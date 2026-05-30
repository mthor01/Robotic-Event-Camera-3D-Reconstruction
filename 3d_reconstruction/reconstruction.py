#!/usr/bin/env python3
"""
Generate depth maps and point clouds for a handful of frames using a trained model.

Loads a checkpoint (e2depth, unet, or mvsnet via the registry), runs inference on
N random frames from a sequence, and saves per-frame outputs:
  - <out_dir>/<frame>/depth_pred.png   — colourised predicted depth
  - <out_dir>/<frame>/depth_gt.png     — colourised GT depth (for comparison)
  - <out_dir>/<frame>/pointcloud.ply   — predicted depth backprojected to 3-D
  - <out_dir>/overview.png             — side-by-side grid of all frames

Usage:
    python3 generate_point_clouds.py \\
        --checkpoint training/checkpoints/e2depth/best.pt \\
        --data_dir   data/real/mult_blocks \\
        --n_frames   5

    # Specific frames instead of random:
    python3 generate_point_clouds.py \\
        --checkpoint training/checkpoints/unet/best.pt \\
        --data_dir   data/real/mult_blocks \\
        --indices 0 50 100 200 400

    # Output directory (default: alongside the checkpoint):
    python3 generate_point_clouds.py \\
        --checkpoint training/checkpoints/e2depth/best.pt \\
        --data_dir   data/real/mult_blocks \\
        --out_dir    results/mult_blocks_e2depth
"""

import sys
import argparse
from pathlib import Path

# Allow imports from training/ and 3d_reconstruction/ roots
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))               # 3d_reconstruction/
sys.path.insert(0, str(_HERE.parent / "training"))  # training/ (models/, etc.)

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import D_MAX, DEPTH_MIN, NUM_BINS, CALIB_DIR as _CALIB_DIR, TRAIN_RESIZE_HW, TRAIN_CROP_HW, \
    TSDF_VOXEL_SIZE, TSDF_SDF_TRUNC_FACTOR, TSDF_DEPTH_MAX
from models import MODEL_REGISTRY
from models.e2depth import linear_normalized_to_depth

# New-style training-script models (no MODEL_REGISTRY; detected by checkpoint keys)
from train_unet import UNet
from train_pose_unet import PoseUNet, project_depth as _project_depth
from train_unet_2 import _pose_to_map, T_SCALE as _T_SCALE, T_SCALE_ABS as _T_SCALE_ABS

CALIB_DIR = _HERE.parent / _CALIB_DIR


# ─────────────────────────────────────────────────────────────────────────────
#  Calibration
# ─────────────────────────────────────────────────────────────────────────────

def load_K(calib_dir: Path, resize_hw: Optional[Tuple[int, int]] = None,
           crop_hw: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int, int]:
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
    """Return T_ee_from_event (4×4): transforms points from event-camera frame
    to end-effector frame.  Composed from T_event_from_rgb and T_rgb_from_ee."""
    T_rgb_from_ee    = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float64)
    T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee
    return np.linalg.inv(T_event_from_ee)  # T_ee_from_event


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ckpt_type(ckpt: dict) -> str:
    """Infer which training script produced a checkpoint from its top-level keys."""
    if "input_mode" in ckpt:                       return "unet_3"
    if "model_s1" in ckpt:                          return "pose_unet"
    if "pose_mode" in ckpt:                          return "unet_2"
    if "model" in ckpt and "config" not in ckpt:    return "unet"
    return "registry"


def load_model(ckpt_path: Path, device: torch.device,
               K: np.ndarray, input_hw: Tuple[int, int]):
    """Instantiate the right model class from the checkpoint's config dict."""
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_type = _detect_ckpt_type(ckpt)

    # ── New-style checkpoints (train_unet / train_unet_2 / train_unet_3 / train_pose_unet) ──
    if ckpt_type == "unet":
        model = UNet(in_ch=ckpt.get("in_ch", NUM_BINS), base=ckpt.get("base", 32)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt, "unet"

    if ckpt_type == "unet_3":
        input_mode = ckpt.get("input_mode", "target_neighbors_pose")
        if input_mode == "target_neighbors_pose":
            raise NotImplementedError(
                "generate_point_clouds.py currently supports unet_3 checkpoints with "
                "input_mode in {'target', 'target_nbins', 'target_neighbors'}. "
                "The warped mode 'target_neighbors_pose' is not supported here yet."
            )
        model = UNet(in_ch=ckpt["in_ch"], base=ckpt.get("base", 32)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt, "unet_3"

    if ckpt_type == "unet_2":
        model = UNet(in_ch=ckpt["in_ch"], base=ckpt.get("base", 32)).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt, "unet_2"

    if ckpt_type == "pose_unet":
        model = PoseUNet(
            num_bins=NUM_BINS,
            window_half=ckpt["window_half"],
            base=ckpt.get("base", 32),
        ).to(device)
        model.s1.load_state_dict(ckpt["model_s1"])
        model.s2.load_state_dict(ckpt["model_s2"])
        model.eval()
        return model, ckpt, "pose_unet"

    # ── Legacy registry-based checkpoints (e2depth, mvsnet, …) ─────────────
    cfg  = ckpt.get("config", {})

    model_name = cfg.get("model", "e2depth")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}' in checkpoint. "
            f"Known: {list(MODEL_REGISTRY.keys())}"
        )

    # Build a minimal args namespace that satisfies each model's build_fn
    import argparse as _ap
    args = _ap.Namespace(
        base          = cfg.get("base",          32),
        num_encoders  = cfg.get("num_encoders",  3),
        num_residuals = cfg.get("num_residuals", 2),
        use_pose_warp = cfg.get("use_pose_warp", False),
        lambda_grad   = cfg.get("lambda_grad",   0.5),
        lambda_smooth = cfg.get("lambda_smooth", 0.01),
        lambda_normal = cfg.get("lambda_normal", 0.1),
        lambda_mean   = cfg.get("lambda_mean",   0.1),
        lambda_mv     = cfg.get("lambda_mv",     0.0),
        depth_min     = cfg.get("depth_min",     DEPTH_MIN),
        depth_max     = cfg.get("depth_max",     D_MAX),
        # MVSNet-specific
        num_views     = cfg.get("num_views",     5),
        num_depth     = cfg.get("num_depth",     192),
        view_interval = cfg.get("view_interval", 5),
    )
    in_channels = cfg.get("in_channels", NUM_BINS)

    build_fn = MODEL_REGISTRY[model_name]
    K_arg    = K      if args.use_pose_warp else None
    hw_arg   = input_hw if args.use_pose_warp else None
    model    = build_fn(args, in_channels, K_arg, hw_arg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, model_name


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def resize_crop(arr: np.ndarray, resize_hw, crop_hw,
                mode: str = "bilinear") -> np.ndarray:
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


def _preprocess_voxels(vox_raw: np.ndarray, resize_hw, crop_hw) -> np.ndarray:
    """Resize/crop a (C, H, W) voxel grid if not already at the target resolution."""
    final_h = crop_hw[0] if crop_hw is not None else (resize_hw[0] if resize_hw is not None else vox_raw.shape[1])
    final_w = crop_hw[1] if crop_hw is not None else (resize_hw[1] if resize_hw is not None else vox_raw.shape[2])
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


def _adapt_voxel_bins_np(vox_np: np.ndarray, n_bins: int) -> np.ndarray:
    """Match voxel channel count to n_bins by truncating or zero-padding."""
    c = int(vox_np.shape[0])
    if c == n_bins:
        return vox_np
    if c > n_bins:
        return vox_np[:n_bins]
    pad = np.zeros((n_bins - c, vox_np.shape[1], vox_np.shape[2]), dtype=vox_np.dtype)
    return np.concatenate([vox_np, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_frame(model, voxel_np: np.ndarray, states,
                  device: torch.device, T_rel=None):
    """Recurrent-model inference (used for legacy registry checkpoints)."""
    vox_t = torch.from_numpy(voxel_np).unsqueeze(0).to(device)
    pred_t, states = model(vox_t, states, T_rel=T_rel)
    return pred_t[0, 0].cpu().numpy(), states


@torch.no_grad()
def _infer_unet(model: UNet, vox_np: np.ndarray,
               device: torch.device) -> np.ndarray:
    """Single-frame inference for train_unet.py checkpoints. Returns (H, W) in [0, 1]."""
    vox = torch.from_numpy(vox_np).unsqueeze(0).to(device)  # (1, C, H, W)
    return model(vox)[0, 0].cpu().numpy()


@torch.no_grad()
def _infer_unet_2(
    model:        UNet,
    vox_t_np:     np.ndarray,   # (C, H, W)
    vox_prev_np:  np.ndarray,
    vox_next_np:  np.ndarray,
    T_world_t:    Optional[np.ndarray],   # (4, 4) or None when pose_mode="none"
    T_world_prev: Optional[np.ndarray],
    T_world_next: Optional[np.ndarray],
    pose_mode:    str,
    device:       torch.device,
) -> np.ndarray:
    """Multi-frame inference for train_unet_2.py checkpoints. Returns (H, W) in [0, 1]."""
    H, W = vox_t_np.shape[1], vox_t_np.shape[2]
    vt = torch.from_numpy(vox_t_np).unsqueeze(0).to(device)
    vp = torch.from_numpy(vox_prev_np).unsqueeze(0).to(device)
    vn = torch.from_numpy(vox_next_np).unsqueeze(0).to(device)

    if pose_mode == "relative" and T_world_t is not None:
        T_tgt_from_world = np.linalg.inv(T_world_t).astype(np.float32)
        T_tgt_from_prev  = (T_tgt_from_world @ T_world_prev).astype(np.float32)
        T_tgt_from_next  = (T_tgt_from_world @ T_world_next).astype(np.float32)
        pp = torch.from_numpy(_pose_to_map(T_tgt_from_prev, H, W)).unsqueeze(0).to(device)
        pn = torch.from_numpy(_pose_to_map(T_tgt_from_next, H, W)).unsqueeze(0).to(device)
        inp = torch.cat([vt, vp, vn, pp, pn], dim=1)
    elif pose_mode == "absolute" and T_world_t is not None:
        pt = torch.from_numpy(_pose_to_map(T_world_t,    H, W, _T_SCALE_ABS)).unsqueeze(0).to(device)
        pp = torch.from_numpy(_pose_to_map(T_world_prev, H, W, _T_SCALE_ABS)).unsqueeze(0).to(device)
        pn = torch.from_numpy(_pose_to_map(T_world_next, H, W, _T_SCALE_ABS)).unsqueeze(0).to(device)
        inp = torch.cat([vt, vp, vn, pt, pp, pn], dim=1)
    else:  # "none" or poses unavailable
        inp = torch.cat([vt, vp, vn], dim=1)

    return model(inp)[0, 0].cpu().numpy()


@torch.no_grad()
def _infer_unet_3(
    model:         UNet,
    vox_t_np:      np.ndarray,   # (C, H, W)
    vox_prev_np:   np.ndarray,
    vox_next_np:   np.ndarray,
    input_mode:    str,
    target_nbins:  int,
    device:        torch.device,
) -> np.ndarray:
    """Inference for train_unet_3.py checkpoints in non-warped input modes."""
    vt = torch.from_numpy(vox_t_np).unsqueeze(0).to(device)
    vp = torch.from_numpy(vox_prev_np).unsqueeze(0).to(device)
    vn = torch.from_numpy(vox_next_np).unsqueeze(0).to(device)

    if input_mode == "target":
        inp = vt
    elif input_mode == "target_nbins":
        inp_np = _adapt_voxel_bins_np(vox_t_np, target_nbins)
        inp = torch.from_numpy(inp_np).unsqueeze(0).to(device)
    elif input_mode == "target_neighbors":
        inp = torch.cat([vt, vp, vn], dim=1)
    else:
        raise ValueError(
            f"Unsupported unet_3 input_mode={input_mode!r}. "
            "Supported modes here: 'target', 'target_nbins', 'target_neighbors'."
        )

    # Debug: Print input shape
    # print(f"    [_infer_unet_3] inp shape: {inp.shape}, input_mode: {input_mode}")
    
    return model(inp)[0, 0].cpu().numpy()


@torch.no_grad()
def _infer_pose_unet(
    model:         PoseUNet,
    vox_t_np:      np.ndarray,        # (C, H, W)
    vox_nbrs:      List[np.ndarray],  # nwin × (C, H, W)
    T_tgt_nbr:     np.ndarray,        # (nwin, 4, 4)
    K:             torch.Tensor,      # (3, 3)
    device:        torch.device,
) -> np.ndarray:
    """Two-stage inference for train_pose_unet.py checkpoints. Returns (H, W) in [0, 1]."""
    vox_t_t   = torch.from_numpy(vox_t_np).unsqueeze(0).to(device)                        # (1, C, H, W)
    vox_nbr_t = torch.from_numpy(np.stack(vox_nbrs)).unsqueeze(0).to(device)              # (1, nwin, C, H, W)
    B, nwin, C, H, W = vox_nbr_t.shape

    # Stage 1: run shared UNet on all frames together
    vox_all    = torch.cat([vox_t_t.unsqueeze(1), vox_nbr_t], dim=1).reshape(B * (nwin + 1), C, H, W)
    dep_s1_all = model.s1(vox_all).reshape(B, nwin + 1, 1, H, W)
    dep_t_s1   = dep_s1_all[:, 0]   # (1, 1, H, W)
    dep_nbr_s1 = dep_s1_all[:, 1:]  # (1, nwin, 1, H, W)

    # Stage 2: project each neighbour's depth into the target frame
    T_tensor  = torch.from_numpy(T_tgt_nbr).to(device=device, dtype=torch.float32)  # (nwin, 4, 4)
    proj_list = [
        _project_depth(dep_nbr_s1[:, ni], T_tensor[ni].unsqueeze(0), K)
        for ni in range(nwin)
    ]  # each (1, 1, H, W)
    dep_proj  = torch.cat(proj_list, dim=1)  # (1, nwin, H, W)

    inp_s2 = torch.cat([vox_t_t, dep_t_s1, dep_proj], dim=1)  # (1, C+1+nwin, H, W)
    return model.s2(inp_s2)[0, 0].cpu().numpy()


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
    frames: list,       # list of (depth_m: np.ndarray H×W float32, T_world_from_cam: np.ndarray 4×4)
    K: np.ndarray,      # (3, 3) camera intrinsics at the depth resolution
    out_path: Path,
    voxel_length: float = 0.004,   # metres per voxel (4 mm)
    sdf_trunc_factor: float = 5.0, # sdf_trunc = voxel_length * factor
    depth_max: float = 0.6,
) -> None:
    """TSDF volumetric fusion of per-frame depth maps into a surface mesh.

    Each depth frame is integrated together with its camera-to-world extrinsic
    so that contributions from different viewpoints are correctly aligned in
    world space.  Requires open3d (pip install open3d).
    """
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
    # Reuse a single black colour image (NoColor mode; required by the API)
    dummy_color = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))

    print(f"  TSDF fusing {len(frames)} depth frames "
          f"(voxel={voxel_length * 1000:.1f} mm, trunc={sdf_trunc * 1000:.1f} mm) …")

    for depth_m, T_world_from_cam in frames:
        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            dummy_color, depth_o3d,
            depth_scale=1.0,       # depth already in metres
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )
        # Open3D integrate expects T_camera_from_world (extrinsic)
        T_cam_from_world = np.linalg.inv(T_world_from_cam)
        volume.integrate(rgbd, intrinsic, T_cam_from_world)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

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
    """Normalise and apply a matplotlib colormap → (H, W, 3) uint8."""
    normed = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba   = getattr(cm, cmap)(normed)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_depth_png(depth_m: np.ndarray, mask: np.ndarray,
                   path: Path, vmin: float, vmax: float) -> None:
    img = colorize(np.where(mask > 0.5, depth_m, np.nan), vmin, vmax)
    # Replace NaN areas with dark grey
    nan_mask = ~(mask > 0.5)
    img[nan_mask] = 40
    plt.imsave(str(path), img)


def save_overview(samples: list, out_path: Path,
                  depth_min: float, depth_max: float) -> None:
    """Save a side-by-side overview PNG: events | GT depth | pred depth | error."""
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5), squeeze=False)
    col_titles = ["Events (summed)", "GT depth", "Pred depth", "Abs error"]

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight="bold")

    for row, s in enumerate(samples):
        ax_ev, ax_gt, ax_pr, ax_er = axes[row]
        frame_idx = s["frame_idx"]

        # Events
        ev_sum = s["voxel"].sum(axis=0) if s["voxel"] is not None else np.zeros_like(s["gt"])
        ev_vis = np.clip((ev_sum - ev_sum.min()) / (ev_sum.max() - ev_sum.min() + 1e-6), 0, 1)
        ax_ev.imshow(ev_vis, cmap="gray")
        ax_ev.set_ylabel(f"frame {frame_idx}", fontsize=8)

        mask = s["mask"]

        # GT
        gt_vis = colorize(np.where(mask > 0.5, s["gt"], 0), depth_min, depth_max)
        gt_vis[~(mask > 0.5)] = 40
        ax_gt.imshow(gt_vis)

        # Pred
        pr_vis = colorize(np.where(mask > 0.5, s["pred"], 0), depth_min, depth_max)
        pr_vis[~(mask > 0.5)] = 40
        ax_pr.imshow(pr_vis)

        # Error
        err = np.abs(s["pred"] - s["gt"]) * (mask > 0.5)
        err_vis = colorize(err, 0, 0.1)   # 0–10 cm range
        err_vis[~(mask > 0.5)] = 40
        im = ax_er.imshow(err_vis)
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

def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps and point clouds for sampled frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Sequence directory (must contain hdf5/ and events/)")
    parser.add_argument("--n_frames", type=int, default=5,
                        help="Number of frames to visualize (overview + per-frame PNGs/PLY)")
    parser.add_argument("--frame_step", type=int, default=None,
                        help="Use every Nth frame from the full sequence for the mesh. "
                             "Defaults to n_total // 20 so ~20 frames are used.")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Explicit visualization frame indices (overrides --n_frames; "
                             "mesh still uses every --frame_step frame)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: <checkpoint_dir>/point_clouds/<seq_name>)")
    parser.add_argument("--num_bins", type=int, default=NUM_BINS,
                        help="Number of event voxel bins")
    parser.add_argument("--depth_min", type=float, default=DEPTH_MIN)
    parser.add_argument("--depth_max", type=float, default=D_MAX)
    parser.add_argument("--resize_h", type=int, default=TRAIN_RESIZE_HW[0],
                        help="Resize height before crop (default: from config.py)")
    parser.add_argument("--resize_w", type=int, default=TRAIN_RESIZE_HW[1],
                        help="Resize width before crop (default: from config.py)")
    parser.add_argument("--crop_h", type=int, default=TRAIN_CROP_HW[0],
                        help="Center-crop height after resize (default: from config.py)")
    parser.add_argument("--crop_w", type=int, default=TRAIN_CROP_HW[1],
                        help="Center-crop width after resize (default: from config.py)")
    parser.add_argument("--warmup", type=int, default=4,
                        help="Number of frames to run before the selected frames "
                             "to warm up recurrent hidden states")
    parser.add_argument("--voxel_size", type=float, default=TSDF_VOXEL_SIZE,
                        help="TSDF voxel side length in metres (default: from config.py)")
    parser.add_argument("--sdf_trunc_factor", type=float, default=TSDF_SDF_TRUNC_FACTOR,
                        help="sdf_trunc = voxel_size * this factor (default: from config.py)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    data_dir  = Path(args.data_dir)
    resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
    crop_hw   = (args.crop_h,   args.crop_w)   if args.crop_h   > 0 and args.crop_w   > 0 else None
    # Mirror training: resize → TRAIN_RESIZE_HW, crop → TRAIN_CROP_HW
    # (args already default to those values, so this branch is only reached when
    #  the user explicitly passes 0 to disable preprocessing)
    if resize_hw is None:
        resize_hw = TRAIN_RESIZE_HW
    if crop_hw is None:
        crop_hw = TRAIN_CROP_HW

    if args.out_dir is None:
        out_dir = ckpt_path.parent / "point_clouds" / data_dir.name
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Camera intrinsics ──────────────────────────────────────────
    K, out_H, out_W = load_K(CALIB_DIR, resize_hw, crop_hw)
    print(f"Input resolution : {out_W}×{out_H}")
    print(f"K:\n{K}")

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────
    model, ckpt_cfg, model_name = load_model(
        ckpt_path, device, K, (out_H, out_W)
    )
    depth_min = ckpt_cfg.get("depth_min", args.depth_min)
    depth_max = ckpt_cfg.get("depth_max", args.depth_max)
    print(f"Model            : {model_name}")
    print(f"Depth range      : {depth_min}m – {depth_max}m")
    
    # ── Debug: show checkpoint config for unet_3 ──
    if model_name == "unet_3":
        print(f"[unet_3 config]")
        print(f"  input_mode   : {ckpt_cfg.get('input_mode', 'N/A')}")
        print(f"  target_nbins : {ckpt_cfg.get('target_nbins', 'N/A')}")
        print(f"  in_ch        : {ckpt_cfg.get('in_ch', 'N/A')}")
        print(f"  stride       : {ckpt_cfg.get('stride', 'N/A')}")
        print(f"  pose_mode    : {ckpt_cfg.get('pose_mode', 'N/A')}")

    if model_name == "mvsnet":
        print("ERROR: mvsnet requires multi-view inputs and is not supported by "
              "this script.  Use reconstruct_gt.py with --model mvsnet instead.")
        return

    # ── Data paths ─────────────────────────────────────────────────
    depth_h5_path   = data_dir / "hdf5" / "depth_in_event_frame.h5"
    voxels_h5_path  = data_dir / "events" / "voxels_cam0.h5"
    spatial_h5_path = data_dir / "hdf5" / "spatial_mask.h5"
    poses_h5_path   = data_dir / "hdf5" / "poses.h5"

    if not depth_h5_path.exists():
        raise FileNotFoundError(f"Missing: {depth_h5_path}")
    if not voxels_h5_path.exists():
        raise FileNotFoundError(
            f"Missing: {voxels_h5_path}\n"
            "Run data_precomputation/precompute_voxels.py first."
        )

    with h5py.File(depth_h5_path, "r") as f:
        n_total = f["depth"].shape[0]
    use_spatial = spatial_h5_path.exists()

    # ── Poses + extrinsics for world-frame point accumulation ──────
    if poses_h5_path.exists():
        with h5py.File(poses_h5_path, "r") as f:
            ee_T_all = f["ee_T"][:].astype(np.float64)  # (N, 4, 4) T_base_from_ee
        T_ee_from_event = load_T_ee_from_event(CALIB_DIR)  # (4, 4)
        use_poses = True
        print(f"Poses loaded     : {len(ee_T_all)} frames — point clouds will be "
              f"transformed to robot base frame")
    else:
        ee_T_all        = None
        T_ee_from_event = None
        use_poses       = False
        print("WARNING: poses.h5 not found — point clouds accumulated in camera "
              "frame only (mesh quality will be poor for multi-frame sequences)")

    # ── Frame selection ────────────────────────────────────────────
    n_viz = args.n_frames
    step  = args.frame_step if args.frame_step is not None else max(1, n_total // 20)

    # Mesh frames: every `step`-th frame across the whole sequence
    mesh_frames = list(range(0, n_total, step))

    if args.indices is not None:
        # Explicit viz frames (must be valid indices)
        viz_frames = sorted([int(i) for i in args.indices if 0 <= int(i) < n_total])
    else:
        # Pick n_viz evenly-spaced frames from mesh_frames for visualization
        if len(mesh_frames) <= n_viz:
            viz_frames = list(mesh_frames)
        else:
            picks      = np.round(np.linspace(0, len(mesh_frames) - 1, n_viz)).astype(int)
            viz_frames = sorted({mesh_frames[i] for i in picks})

    viz_set  = set(viz_frames)
    mesh_set = set(mesh_frames)

    # Warm-up frames: run all frames consecutively from warmup_start through
    # to the last mesh frame so the ConvLSTM recurrent state is always current.
    # (Skipping frames between mesh indices would leave stale hidden states.)
    warmup_start = max(0, mesh_frames[0] - args.warmup)
    all_frames   = list(range(warmup_start, mesh_frames[-1] + 1))

    print(f"Total frames     : {n_total}")
    print(f"Frame step       : {step}  ({len(mesh_frames)} mesh frames)")
    print(f"Viz   frames ({len(viz_frames):3d}): {viz_frames}")
    print(f"Warmup from      : {warmup_start}")

    # ── Inference ─────────────────────────────────────────────────
    depth_f   = h5py.File(depth_h5_path,   "r")
    voxels_f  = h5py.File(voxels_h5_path,  "r")
    spatial_f = h5py.File(spatial_h5_path, "r") if use_spatial else None

    samples      = []
    mesh_data    = []   # (masked pred depth, T_world_from_cam) per mesh frame
    gt_mesh_data = []   # (masked GT depth,   T_world_from_cam) per mesh frame
    # ── Inference loop setup ──────────────────────────────────────────────────
    _stateless = model_name in ("unet", "unet_2", "unet_3", "pose_unet")
    if _stateless:
        _sv  = ckpt_cfg.get("stride", 3)          # frame stride (unet_2)
        _wh  = ckpt_cfg.get("window_half", 2)     # window half (pose_unet)
        _ws  = ckpt_cfg.get("window_stride", 3)   # window stride (pose_unet)
        _margin = 0
        if model_name == "pose_unet":
            _margin = _wh * _ws
        elif model_name == "unet_2":
            _margin = _sv
        elif model_name == "unet_3":
            _im = ckpt_cfg.get("input_mode", "target_neighbors_pose")
            _margin = _sv if _im in ("target_neighbors", "target_neighbors_pose") else 0
        if _margin > 0:
            print(f"Frame margin     : ±{_margin} frames (boundary frames skipped)")
        run_frames = sorted(fi for fi in mesh_set | viz_set
                            if _margin <= fi < n_total - _margin)
        K_tensor   = torch.from_numpy(K.astype(np.float32)).to(device)
    else:
        run_frames = all_frames

    states = None

    try:
        for frame_idx in run_frames:
            vox_raw  = voxels_f["voxels"][frame_idx].astype(np.float32)
            vox_t_np = _preprocess_voxels(vox_raw, resize_hw, crop_hw)

            # ── Per-frame prediction ─────────────────────────────────────────
            if model_name == "unet":
                pred_norm = _infer_unet(model, vox_t_np, device)

            elif model_name == "unet_2":
                _pm = ckpt_cfg.get("pose_mode", "relative")
                _T_wt = _T_wp = _T_wn = None
                if use_poses and _pm != "none":
                    _T_wt = (ee_T_all[frame_idx]        @ T_ee_from_event).astype(np.float32)
                    _T_wp = (ee_T_all[frame_idx - _sv]  @ T_ee_from_event).astype(np.float32)
                    _T_wn = (ee_T_all[frame_idx + _sv]  @ T_ee_from_event).astype(np.float32)
                elif _pm != "none":
                    print(f"WARNING: pose_mode={_pm!r} but poses.h5 missing — falling back to 'none'")
                    _pm = "none"
                vox_prev_np = _preprocess_voxels(
                    voxels_f["voxels"][frame_idx - _sv].astype(np.float32), resize_hw, crop_hw)
                vox_next_np = _preprocess_voxels(
                    voxels_f["voxels"][frame_idx + _sv].astype(np.float32), resize_hw, crop_hw)
                pred_norm = _infer_unet_2(model, vox_t_np, vox_prev_np, vox_next_np,
                                          _T_wt, _T_wp, _T_wn, _pm, device)

            elif model_name == "unet_3":
                _im = ckpt_cfg.get("input_mode", "target_neighbors_pose")
                _tn = int(ckpt_cfg.get("target_nbins", ckpt_cfg.get("in_ch", NUM_BINS)))

                if _im in ("target_neighbors", "target_neighbors_pose"):
                    vox_prev_np = _preprocess_voxels(
                        voxels_f["voxels"][frame_idx - _sv].astype(np.float32), resize_hw, crop_hw)
                    vox_next_np = _preprocess_voxels(
                        voxels_f["voxels"][frame_idx + _sv].astype(np.float32), resize_hw, crop_hw)
                else:
                    # Unused for target-only modes, but keep shape-compatible placeholders.
                    vox_prev_np = vox_t_np
                    vox_next_np = vox_t_np

                # Debug: print input shapes on first frame
                if frame_idx == (min(viz_frames) if viz_frames else 0):
                    print(f"[unet_3 inference @ frame {frame_idx}]")
                    print(f"  input_mode   : {_im}")
                    print(f"  target_nbins : {_tn}")
                    print(f"  vox_t shape  : {vox_t_np.shape}")
                    print(f"  vox_p shape  : {vox_prev_np.shape}")
                    print(f"  vox_n shape  : {vox_next_np.shape}")

                pred_norm = _infer_unet_3(
                    model,
                    vox_t_np,
                    vox_prev_np,
                    vox_next_np,
                    _im,
                    _tn,
                    device,
                )

            elif model_name == "pose_unet":
                offsets  = [o for o in range(-_wh, _wh + 1) if o != 0]
                vox_nbrs = [_preprocess_voxels(
                    voxels_f["voxels"][frame_idx + o * _ws].astype(np.float32),
                    resize_hw, crop_hw) for o in offsets]
                if use_poses:
                    _T_wt_mat = (ee_T_all[frame_idx] @ T_ee_from_event).astype(np.float32)
                    _T_tgt_w  = np.linalg.inv(_T_wt_mat).astype(np.float32)
                    T_tgt_nbr_arr = np.stack([
                        (_T_tgt_w @ (ee_T_all[frame_idx + o * _ws] @ T_ee_from_event).astype(np.float32))
                        for o in offsets
                    ]).astype(np.float32)
                else:
                    T_tgt_nbr_arr = np.eye(4, dtype=np.float32)[None].repeat(len(offsets), axis=0)
                pred_norm = _infer_pose_unet(model, vox_t_np, vox_nbrs,
                                             T_tgt_nbr_arr, K_tensor, device)

            else:
                # Legacy recurrent model: needs sequential warmup pass
                pred_norm, states = predict_frame(model, vox_t_np, states, device)
                if frame_idx not in mesh_set:
                    continue   # warmup / state-update frame — skip post-processing

            # Convert normalised prediction → metres
            pred_m = depth_min + pred_norm * (depth_max - depth_min)

            # GT depth + mask — skip resize_crop if already at final resolution
            gt = depth_f["depth"][frame_idx].astype(np.float32)
            if gt.shape[0] != out_H or gt.shape[1] != out_W:
                gt = resize_crop(gt, resize_hw, crop_hw, mode="bilinear")

            mask = ((gt > depth_min) & (gt < depth_max)).astype(np.float32)
            if use_spatial:
                sp = spatial_f["mask"][frame_idx].astype(np.float32)
                if sp.shape[0] != out_H or sp.shape[1] != out_W:
                    sp = resize_crop(sp, resize_hw, crop_hw, mode="nearest")
                mask[sp < 0.5] = 0.0

            # Per-frame camera-space point cloud (used for the per-frame PLY only)
            pts = depth_to_pointcloud(pred_m, mask, K)

            # Accumulate masked depth frames + extrinsics for TSDF fusion.
            # Zero out pixels outside the mask so they are treated as invalid
            # (depth == 0 is skipped by Open3D's TSDF integrator).
            if use_poses and frame_idx < len(ee_T_all):
                T_base_from_event = ee_T_all[frame_idx] @ T_ee_from_event
                pred_masked = np.where(mask > 0.5, pred_m, np.float32(0.0))
                gt_masked   = np.where(mask > 0.5, gt,     np.float32(0.0))
                mesh_data.append((pred_masked, T_base_from_event))
                gt_mesh_data.append((gt_masked,   T_base_from_event))

            if frame_idx not in viz_set:
                continue   # mesh-only frame — skip per-frame outputs

            # ── Per-frame directory (visualization frames only) ────
            frame_dir = out_dir / f"frame_{frame_idx:05d}"
            frame_dir.mkdir(exist_ok=True)

            # Depth PNGs
            save_depth_png(pred_m, mask, frame_dir / "depth_pred.png",
                           depth_min, depth_max)
            save_depth_png(gt,     mask, frame_dir / "depth_gt.png",
                           depth_min, depth_max)

            # Per-frame point cloud
            ply_path = frame_dir / "pointcloud.ply"
            save_ply(pts, ply_path)

            mae = float(np.abs(pred_m - gt)[mask > 0.5].mean()) if (mask > 0.5).any() else float("nan")
            print(f"  frame {frame_idx:5d} | {len(pts):6d} pts | MAE {mae * 100:.2f} cm | → {frame_dir.name}/")

            samples.append({
                "frame_idx": frame_idx,
                "gt":        gt,
                "pred":      pred_m,
                "mask":      mask,
                "voxel":     vox_t_np,
            })
    finally:
        depth_f.close()
        voxels_f.close()
        if spatial_f is not None:
            spatial_f.close()

    # ── 3-D mesh via TSDF fusion ──────────────────────────────────
    if mesh_data:
        tsdf_fuse(mesh_data, K, out_dir / "mesh.obj",
                  voxel_length=args.voxel_size,
                  sdf_trunc_factor=args.sdf_trunc_factor,
                  depth_max=depth_max)
    if gt_mesh_data:
        tsdf_fuse(gt_mesh_data, K, out_dir / "gt_mesh.obj",
                  voxel_length=args.voxel_size,
                  sdf_trunc_factor=args.sdf_trunc_factor,
                  depth_max=depth_max)

    # ── Overview PNG (visualization frames only) ──────────────────
    if samples:
        save_overview(samples, out_dir / "overview.png", depth_min, depth_max)

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()
