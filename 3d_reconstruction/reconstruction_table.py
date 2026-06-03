#!/usr/bin/env python3
"""
Depth-map / point-cloud reconstruction for unet_table checkpoints only.

Identical workflow to reconstruction.py but uses the precomputed table-plane
channel stored in hdf5/table_plane.h5 (produced by
data_precomputation/precompute_table_plane.py) instead of computing it on-the-fly.

Per-frame outputs written to <out_dir>/:
  frame_NNNNN/depth_pred.png   — colourised predicted depth
  frame_NNNNN/depth_gt.png     — colourised GT depth
  frame_NNNNN/pointcloud.ply   — predicted depth backprojected to 3-D
  overview.png                 — side-by-side grid of all viz frames
  mesh.obj                     — TSDF-fused predicted mesh
  gt_mesh.obj                  — TSDF-fused GT mesh

Usage:
    python3 reconstruction_table.py \\
        --checkpoint training/checkpoints/unet_table/best_myrun.pth \\
        --data_dir   data/real/lego_1

    python3 reconstruction_table.py \\
        --checkpoint training/checkpoints/unet_table/best_myrun.pth \\
        --data_dir   data/real/lego_1 \\
        --indices 0 50 100 200 400 \\
        --out_dir results/lego_1_table
"""

import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "training"))

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import (
    D_MAX, DEPTH_MIN, NUM_BINS,
    CALIB_DIR as _CALIB_DIR,
    TRAIN_RESIZE_HW, TRAIN_CROP_HW,
    TSDF_VOXEL_SIZE, TSDF_SDF_TRUNC_FACTOR, TSDF_DEPTH_MAX,
    SPATIAL_CUBE_SIDE, SPATIAL_TARGET_X, SPATIAL_TARGET_Y, SPATIAL_TARGET_Z,
)
from train_unet import UNet

CALIB_DIR = _HERE / _CALIB_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_K(
    calib_dir: Path,
    resize_hw: Optional[Tuple[int, int]] = None,
    crop_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, int, int]:
    """Return camera intrinsics scaled to the model's input resolution."""
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K = ev["camera_matrix"].copy().astype(np.float64)
    native_W = int(ev["image_size"][0])
    native_H = int(ev["image_size"][1])
    cur_H, cur_W = native_H, native_W

    if resize_hw is not None:
        rh, rw = resize_hw
        K[0] *= rw / cur_W
        K[1] *= rh / cur_H
        cur_H, cur_W = rh, rw

    if crop_hw is not None:
        ch, cw = crop_hw
        K[0, 2] -= (cur_W - cw) // 2
        K[1, 2] -= (cur_H - ch) // 2
        cur_H, cur_W = ch, cw

    return K, cur_H, cur_W


def load_T_ee_from_event(calib_dir: Path) -> np.ndarray:
    """Return T_ee_from_event (4×4) composed from T_event_from_rgb and T_rgb_from_ee."""
    T_rgb_from_ee    = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float64)
    T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee
    return np.linalg.inv(T_event_from_ee)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading  (unet_table only)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> Tuple[UNet, dict]:
    """Load a unet_table checkpoint and return (model, ckpt_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Sanity-check: must look like a unet_table checkpoint
    is_unet_table = (
        "table_z" in ckpt
        or ("model" in ckpt and "config" not in ckpt and ckpt.get("in_ch", NUM_BINS) > NUM_BINS)
    )
    if not is_unet_table:
        raise ValueError(
            f"Checkpoint does not appear to be a unet_table checkpoint: {ckpt_path}\n"
            "Expected keys: 'model', 'table_z' (or 'in_ch' > NUM_BINS)."
        )

    in_ch = ckpt.get("in_ch", NUM_BINS + 1)
    base  = ckpt.get("base",  32)
    model = UNet(in_ch=in_ch, base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def resize_crop(
    arr: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
    mode: str = "bilinear",
) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if resize_hw is not None:
        kw = {} if mode == "nearest" else {"align_corners": False}
        t = F.interpolate(t, size=resize_hw, mode=mode, **kw)
    if crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    return t[0, 0].numpy()


def _preprocess_voxels(
    vox_raw: np.ndarray,
    resize_hw: Optional[Tuple[int, int]],
    crop_hw:   Optional[Tuple[int, int]],
) -> np.ndarray:
    """Resize/crop a (C, H, W) voxel grid to the target resolution."""
    final_h = (crop_hw[0]   if crop_hw   is not None else
               resize_hw[0] if resize_hw is not None else vox_raw.shape[1])
    final_w = (crop_hw[1]   if crop_hw   is not None else
               resize_hw[1] if resize_hw is not None else vox_raw.shape[2])
    if vox_raw.shape[1] == final_h and vox_raw.shape[2] == final_w:
        return vox_raw
    t = torch.from_numpy(vox_raw).unsqueeze(0)  # (1, C, H, W)
    if resize_hw is not None:
        t = F.interpolate(t, size=resize_hw, mode="bilinear", align_corners=False)
    if crop_hw is not None:
        ch, cw = crop_hw
        y0 = (t.shape[2] - ch) // 2
        x0 = (t.shape[3] - cw) // 2
        t = t[:, :, y0:y0 + ch, x0:x0 + cw]
    return t.squeeze(0).numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _infer(
    model:    UNet,
    vox_np:   np.ndarray,   # (C, H, W)  preprocessed voxels
    tbl_np:   np.ndarray,   # (H_tbl, W_tbl) precomputed table-plane channel [0, 1]
    device:   torch.device,
) -> np.ndarray:
    """Run a single unet_table forward pass.  Returns (H, W) normalised depth in [0, 1]."""
    vox_H, vox_W = vox_np.shape[1], vox_np.shape[2]

    # Resize table-plane channel to voxel resolution if needed
    tbl_t = torch.from_numpy(tbl_np.astype(np.float32)).unsqueeze(0)  # (1, H_tbl, W_tbl)
    if tbl_t.shape[-2] != vox_H or tbl_t.shape[-1] != vox_W:
        tbl_t = F.interpolate(
            tbl_t.unsqueeze(0), (vox_H, vox_W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

    tbl_np_r = tbl_t.numpy()  # (1, H, W) — will be concatenated below

    inp_np = np.concatenate([vox_np, tbl_np_r], axis=0)   # (C+1, H, W)
    inp    = torch.from_numpy(inp_np).unsqueeze(0).to(device)   # (1, C+1, H, W)
    return model(inp)[0, 0].cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Point-cloud helpers
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(depth_m: np.ndarray, mask: np.ndarray,
                        K: np.ndarray) -> np.ndarray:
    """Backproject masked depth (H, W) → (N, 3) XYZ point cloud in camera frame."""
    H, W = depth_m.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    valid = (mask > 0.5) & (depth_m > 0)
    z = depth_m[valid]
    x = (xx[valid] - K[0, 2]) * z / K[0, 0]
    y = (yy[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def save_ply(pts: np.ndarray, path: Path) -> None:
    """Write (N, 3) float32 XYZ as binary PLY."""
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(pts.tobytes())


def tsdf_fuse(
    frames:           list,
    K:                np.ndarray,
    out_path:         Path,
    voxel_length:     float = 0.004,
    sdf_trunc_factor: float = 5.0,
    depth_max:        float = 0.6,
    cube_center:      Optional[np.ndarray] = None,
    cube_half_side:   float = 0.0,
) -> None:
    """TSDF volumetric fusion of per-frame depth maps into a surface mesh."""
    try:
        import open3d as o3d
    except ImportError:
        print("  [mesh] open3d not available — skipping. Install: pip install open3d")
        return

    if not frames:
        print("  [mesh] No frames to fuse — skipping")
        return

    H, W = frames[0][0].shape
    sdf_trunc = voxel_length * sdf_trunc_factor
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H,
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    dummy_color = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))

    print(f"  TSDF fusing {len(frames)} depth frames "
          f"(voxel={voxel_length * 1000:.1f} mm, trunc={sdf_trunc * 1000:.1f} mm) …")

    for depth_m, T_world_from_cam in frames:
        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy_color, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        T_cam_from_world = np.linalg.inv(T_world_from_cam)
        volume.integrate(rgbd, intrinsic, T_cam_from_world)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    if cube_center is not None and cube_half_side > 0.0:
        verts  = np.asarray(mesh.vertices)
        diff   = np.abs(verts - cube_center[None, :])
        inside = np.where(np.all(diff <= cube_half_side, axis=1))[0]
        mesh   = mesh.select_by_index(inside)
        mesh.compute_vertex_normals()
        print(f"  Cube crop (side={cube_half_side*2*100:.0f} cm) → "
              f"{len(np.asarray(mesh.vertices)):,} verts, "
              f"{len(np.asarray(mesh.triangles)):,} triangles")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh, write_vertex_normals=True)
    n_v = len(np.asarray(mesh.vertices))
    n_t = len(np.asarray(mesh.triangles))
    print(f"  Mesh → {out_path}  ({n_v:,} verts, {n_t:,} triangles)")


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def colorize(arr: np.ndarray, vmin: float, vmax: float,
             cmap: str = "turbo") -> np.ndarray:
    normed = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba   = getattr(cm, cmap)(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_depth_png(depth_m: np.ndarray, mask: np.ndarray,
                   path: Path, vmin: float, vmax: float) -> None:
    img = colorize(np.where(mask > 0.5, depth_m, np.nan), vmin, vmax)
    img[~(mask > 0.5)] = 40
    plt.imsave(str(path), img)


def save_overview(samples: list, out_path: Path,
                  depth_min: float, depth_max: float) -> None:
    """Side-by-side: events | GT depth | pred depth | abs error."""
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5), squeeze=False)
    col_titles = ["Events (summed)", "GT depth", "Pred depth", "Abs error"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for row, s in enumerate(samples):
        ax_ev, ax_gt, ax_pr, ax_er = axes[row]
        frame_idx = s["frame_idx"]

        ev_sum = s["voxel"].sum(axis=0)
        ev_vis = np.clip((ev_sum - ev_sum.min()) / (ev_sum.max() - ev_sum.min() + 1e-6), 0, 1)
        ax_ev.imshow(ev_vis, cmap="gray")
        ax_ev.set_ylabel(f"frame {frame_idx}", fontsize=8)

        mask = s["mask"]
        gt_vis = colorize(np.where(mask > 0.5, s["gt"],   0), depth_min, depth_max)
        pr_vis = colorize(np.where(mask > 0.5, s["pred"], 0), depth_min, depth_max)
        gt_vis[~(mask > 0.5)] = 40
        pr_vis[~(mask > 0.5)] = 40
        ax_gt.imshow(gt_vis)
        ax_pr.imshow(pr_vis)

        err = np.abs(s["pred"] - s["gt"]) * (mask > 0.5)
        err_vis = colorize(err, 0, 0.1)
        err_vis[~(mask > 0.5)] = 40
        ax_er.imshow(err_vis)
        ax_er.set_xlabel(
            f"MAE {err[mask > 0.5].mean() * 100:.1f} cm" if (mask > 0.5).any() else "",
            fontsize=8,
        )

        for ax in (ax_ev, ax_gt, ax_pr, ax_er):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overview → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth-map / point-cloud reconstruction for unet_table checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a unet_table checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Sequence directory (must contain hdf5/ and events/)")
    parser.add_argument("--n_frames", type=int, default=5,
                        help="Number of frames to visualise in the overview")
    parser.add_argument("--frame_step", type=int, default=None,
                        help="Use every Nth frame for the mesh (default: n_total // 20)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Explicit visualisation frame indices (overrides --n_frames)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: <ckpt_dir>/point_clouds/<seq_name>)")
    parser.add_argument("--depth_min", type=float, default=DEPTH_MIN)
    parser.add_argument("--depth_max", type=float, default=D_MAX)
    parser.add_argument("--resize_h", type=int, default=TRAIN_RESIZE_HW[0])
    parser.add_argument("--resize_w", type=int, default=TRAIN_RESIZE_HW[1])
    parser.add_argument("--crop_h",   type=int, default=TRAIN_CROP_HW[0])
    parser.add_argument("--crop_w",   type=int, default=TRAIN_CROP_HW[1])
    parser.add_argument("--voxel_size",        type=float, default=TSDF_VOXEL_SIZE)
    parser.add_argument("--sdf_trunc_factor",  type=float, default=TSDF_SDF_TRUNC_FACTOR)
    parser.add_argument("--cube_side", type=float, default=SPATIAL_CUBE_SIDE)
    parser.add_argument("--target_x", type=float, default=SPATIAL_TARGET_X)
    parser.add_argument("--target_y", type=float, default=SPATIAL_TARGET_Y)
    parser.add_argument("--target_z", type=float, default=SPATIAL_TARGET_Z)
    args = parser.parse_args()

    ckpt_path      = Path(args.checkpoint)
    data_dir       = Path(args.data_dir)
    cube_center    = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
    cube_half_side = args.cube_side / 2.0

    resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else TRAIN_RESIZE_HW
    crop_hw   = (args.crop_h,   args.crop_w)   if args.crop_h   > 0 and args.crop_w   > 0 else TRAIN_CROP_HW

    if args.out_dir is None:
        out_dir = ckpt_path.parent / "point_clouds" / data_dir.name
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data paths ────────────────────────────────────────────────
    depth_h5_path      = data_dir / "hdf5" / "depth_in_event_frame.h5"
    voxels_h5_path     = data_dir / "events" / "voxels_cam0.h5"
    poses_h5_path      = data_dir / "hdf5" / "poses.h5"
    table_plane_h5_path = data_dir / "hdf5" / "table_plane.h5"

    for p, hint in [
        (depth_h5_path,       "Run data_precomputation/project_realsense_to_event.py first."),
        (voxels_h5_path,      "Run data_precomputation/precompute_voxels.py first."),
        (table_plane_h5_path, "Run data_precomputation/precompute_table_plane.py first."),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\n{hint}")

    # ── Camera intrinsics ─────────────────────────────────────────
    K, out_H, out_W = load_K(CALIB_DIR, resize_hw, crop_hw)
    print(f"Input resolution : {out_W}×{out_H}")

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")

    # ── Model ─────────────────────────────────────────────────────
    model, ckpt = load_model(ckpt_path, device)
    depth_min = ckpt.get("depth_min", args.depth_min)
    depth_max = ckpt.get("depth_max", args.depth_max)
    table_z   = ckpt.get("table_z",   None)

    print(f"Checkpoint       : {ckpt_path.name}")
    print(f"  in_ch          : {ckpt.get('in_ch', NUM_BINS + 1)}")
    print(f"  table_z        : {table_z} m")
    print(f"  depth range    : {depth_min} – {depth_max} m")

    # Read table_z from the precomputed h5 as a cross-check
    with h5py.File(table_plane_h5_path, "r") as _tf:
        precomp_table_z = float(_tf.attrs.get("table_z_m", 0.0))
    if table_z is not None and abs(table_z - precomp_table_z) > 1e-4:
        print(f"  WARNING: checkpoint table_z ({table_z:.4f} m) differs from "
              f"precomputed table_z ({precomp_table_z:.4f} m). "
              "Using precomputed value for channel loading; model was trained with "
              "the checkpoint value.")

    # ── Poses ─────────────────────────────────────────────────────
    if poses_h5_path.exists():
        with h5py.File(poses_h5_path, "r") as f:
            ee_T_all = f["ee_T"][:].astype(np.float64)   # (N, 4, 4)
        T_ee_from_event = load_T_ee_from_event(CALIB_DIR)
        use_poses = True
        print(f"Poses loaded     : {len(ee_T_all)} frames")
    else:
        ee_T_all        = None
        T_ee_from_event = None
        use_poses       = False
        print("WARNING: poses.h5 not found — TSDF mesh will be in camera frame only")

    # ── Frame counts ──────────────────────────────────────────────
    with h5py.File(depth_h5_path, "r") as f:
        n_total = f["depth"].shape[0]
    with h5py.File(table_plane_h5_path, "r") as f:
        n_tbl = f["table_plane"].shape[0]

    n_total = min(n_total, n_tbl)   # don't go past what's precomputed

    # ── Frame selection ────────────────────────────────────────────
    step  = args.frame_step if args.frame_step is not None else max(1, n_total // 20)
    mesh_frames = list(range(0, n_total, step))

    if args.indices is not None:
        viz_frames = sorted([int(i) for i in args.indices if 0 <= int(i) < n_total])
    else:
        if len(mesh_frames) <= args.n_frames:
            viz_frames = list(mesh_frames)
        else:
            picks      = np.round(np.linspace(0, len(mesh_frames) - 1, args.n_frames)).astype(int)
            viz_frames = sorted({mesh_frames[i] for i in picks})

    run_frames = sorted(set(mesh_frames) | set(viz_frames))

    print(f"Total frames     : {n_total}")
    print(f"Frame step       : {step}  ({len(mesh_frames)} mesh frames)")
    print(f"Viz frames ({len(viz_frames):3d}) : {viz_frames}")

    # ── Inference ─────────────────────────────────────────────────
    depth_f  = h5py.File(depth_h5_path,       "r")
    voxels_f = h5py.File(voxels_h5_path,      "r")
    table_f  = h5py.File(table_plane_h5_path, "r")

    samples      = []
    mesh_data    = []
    gt_mesh_data = []
    viz_set      = set(viz_frames)
    mesh_set     = set(mesh_frames)

    try:
        for frame_idx in run_frames:
            # Voxels
            vox_raw  = voxels_f["voxels"][frame_idx].astype(np.float32)
            vox_np   = _preprocess_voxels(vox_raw, resize_hw, crop_hw)

            # Table-plane channel (precomputed, native camera resolution)
            tbl_np   = table_f["table_plane"][frame_idx].astype(np.float32)

            pred_norm = _infer(model, vox_np, tbl_np, device)
            pred_m    = depth_min + pred_norm * (depth_max - depth_min)

            # GT depth
            gt = depth_f["depth"][frame_idx].astype(np.float32)
            if gt.shape[0] != out_H or gt.shape[1] != out_W:
                gt = resize_crop(gt, resize_hw, crop_hw, mode="bilinear")
            mask = ((gt > depth_min) & (gt < depth_max)).astype(np.float32)

            # Per-frame point cloud
            pts = depth_to_pointcloud(pred_m, mask, K)

            # Accumulate for TSDF
            if use_poses and frame_idx < len(ee_T_all) and frame_idx in mesh_set:
                T_base_from_event = ee_T_all[frame_idx] @ T_ee_from_event
                pred_masked = np.where(mask > 0.5, pred_m, np.float32(0.0))
                gt_masked   = np.where(mask > 0.5, gt,     np.float32(0.0))
                mesh_data.append((pred_masked,  T_base_from_event))
                gt_mesh_data.append((gt_masked, T_base_from_event))

            if frame_idx not in viz_set:
                continue

            frame_dir = out_dir / f"frame_{frame_idx:05d}"
            frame_dir.mkdir(exist_ok=True)

            save_depth_png(pred_m, mask, frame_dir / "depth_pred.png", depth_min, depth_max)
            save_depth_png(gt,     mask, frame_dir / "depth_gt.png",   depth_min, depth_max)
            save_ply(pts, frame_dir / "pointcloud.ply")

            mae = float(np.abs(pred_m - gt)[mask > 0.5].mean()) if (mask > 0.5).any() else float("nan")
            print(f"  frame {frame_idx:5d} | {len(pts):6d} pts | MAE {mae * 100:.2f} cm | → {frame_dir.name}/")

            samples.append({
                "frame_idx": frame_idx,
                "gt":        gt,
                "pred":      pred_m,
                "mask":      mask,
                "voxel":     vox_np,
            })
    finally:
        depth_f.close()
        voxels_f.close()
        table_f.close()

    # ── TSDF mesh ─────────────────────────────────────────────────
    if mesh_data:
        tsdf_fuse(mesh_data, K, out_dir / "mesh.obj",
                  voxel_length=args.voxel_size,
                  sdf_trunc_factor=args.sdf_trunc_factor,
                  depth_max=depth_max,
                  cube_center=cube_center,
                  cube_half_side=cube_half_side)
    if gt_mesh_data:
        tsdf_fuse(gt_mesh_data, K, out_dir / "gt_mesh.obj",
                  voxel_length=args.voxel_size,
                  sdf_trunc_factor=args.sdf_trunc_factor,
                  depth_max=depth_max,
                  cube_center=cube_center,
                  cube_half_side=cube_half_side)

    # ── Overview PNG ──────────────────────────────────────────────
    if samples:
        save_overview(samples, out_dir / "overview.png", depth_min, depth_max)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
