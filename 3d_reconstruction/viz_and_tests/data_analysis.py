#!/usr/bin/env python3
"""
Data analysis script — generates image plots from recording data.

Subcommands:
  poses         – Camera poses with RGB images (3-D scatter + image grid)
  turning_point – Frames around the sharpest EE direction reversal
  all           – Run both analyses above

Output images are saved to  <data_dir>/data_plots/  by default.

Usage examples:
    python3 data_analysis.py poses         --data_dir data/real/1
    python3 data_analysis.py turning_point --data_dir data/real/1
    python3 data_analysis.py all           --data_dir data/real/1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import POSE_VIZ_AXIS_LEN, POSE_VIZ_ARROW_LEN


# ══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════

def ensure_out_dir(data_dir: Path) -> Path:
    """Return <data_dir>/data_plots, creating it if necessary."""
    d = data_dir / "data_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
#  1. poses
# ══════════════════════════════════════════════════════════════════════

def run_poses(seq_dir: Path, out_dir: Path, calib_dir: str = "camera_data",
              n_samples: int = 6, seed: Optional[int] = None) -> None:
    """Visualize camera poses + corresponding RGB images."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.gridspec import GridSpec

    calib_dir_p = Path(calib_dir)
    AXIS_LEN = POSE_VIZ_AXIS_LEN; ARROW_LEN = POSE_VIZ_ARROW_LEN

    T_rgb_from_ee = np.load(calib_dir_p / "T_rgb_from_ee.npz")["T"].astype(np.float64)
    T_ee_rgb = np.linalg.inv(T_rgb_from_ee)

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]
    n_frames = ee_Ts.shape[0]
    n_samples = min(n_samples, n_frames)

    # Farthest-point sampling
    all_pos = ee_Ts[:, :3, 3]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n_frames))
    selected = [first]
    min_dists = np.full(n_frames, np.inf)
    diff = all_pos - all_pos[first]
    min_dists = np.minimum(min_dists, (diff * diff).sum(axis=1))
    for _ in range(n_samples - 1):
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        diff = all_pos - all_pos[farthest]
        min_dists = np.minimum(min_dists, (diff * diff).sum(axis=1))
    sampled_indices = sorted(selected)

    rgb_Ts = np.stack([ee_Ts[i] @ T_ee_rgb for i in sampled_indices], axis=0)
    positions = rgb_Ts[:, :3, 3]
    rotations = rgb_Ts[:, :3, :3]

    with h5py.File(seq_dir / "hdf5" / "realsense.h5", "r") as f:
        rgb_frames = [f["rgb"][i] for i in sampled_indices]

    img_cols = 2
    img_rows = int(np.ceil(n_samples / img_cols))
    fig = plt.figure(figsize=(7 + img_cols * 3.2, max(6, img_rows * 3.2)))
    gs = GridSpec(img_rows, 1 + img_cols, figure=fig, width_ratios=[2.5] + [1] * img_cols,
                  wspace=0.05, hspace=0.35)

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=60, c="steelblue", zorder=5)
    for k, (pos, rot, fi) in enumerate(zip(positions, rotations, sampled_indices)):
        for i, col in enumerate(["red", "green", "blue"]):
            d = rot[:, i] * AXIS_LEN
            ax3d.quiver(pos[0], pos[1], pos[2], d[0], d[1], d[2], color=col, linewidth=1.2, arrow_length_ratio=0.3)
        z_dir = rot[:, 2] * ARROW_LEN
        ax3d.quiver(pos[0], pos[1], pos[2], z_dir[0], z_dir[1], z_dir[2],
                    color="black", linewidth=1.5, arrow_length_ratio=0.25, alpha=0.7)
        ax3d.text(pos[0], pos[1], pos[2], f"  #{k}", fontsize=8, color="steelblue")
    ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="steelblue", lw=0.8, alpha=0.5)
    scene_centre = positions.mean(axis=0)
    ax3d.scatter(*scene_centre, s=120, marker="*", c="red", zorder=10)
    ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.set_title(f"RGB camera poses ({seq_dir.name})", fontsize=9)
    ranges = positions.max(axis=0) - positions.min(axis=0)
    max_range = ranges.max() / 2 if ranges.max() > 0 else 0.1
    mid = (positions.max(axis=0) + positions.min(axis=0)) / 2
    ax3d.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax3d.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax3d.set_zlim(mid[2] - max_range, mid[2] + max_range)

    for k, (rgb, fi) in enumerate(zip(rgb_frames, sampled_indices)):
        row, col = k // img_cols, k % img_cols
        ax_img = fig.add_subplot(gs[row, 1 + col])
        ax_img.imshow(rgb); ax_img.set_title(f"#{k} [idx {fi}]", fontsize=8); ax_img.axis("off")
    for k in range(n_samples, img_rows * img_cols):
        fig.add_subplot(gs[k // img_cols, 1 + k % img_cols]).axis("off")

    fig.suptitle(f"Camera poses — {seq_dir.name}  ({n_samples} FPS)", fontsize=11, fontweight="bold", y=1.01)
    out = out_dir / "poses_viz.png"
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  2. turning_point
# ══════════════════════════════════════════════════════════════════════

def _direction_change_angles(positions: np.ndarray, smooth_k: int = 1) -> np.ndarray:
    pos = box_smooth(positions[:, 0], smooth_k)[:, None]
    for ax in [1, 2]:
        pos = np.hstack([pos, box_smooth(positions[:, ax], smooth_k)[:, None]])
    vel = np.diff(pos, axis=0)
    n = len(positions)
    angles = np.zeros(n)
    for i in range(1, n - 1):
        v_in, v_out = vel[i - 1], vel[i]
        ni, no = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if ni < 1e-9 or no < 1e-9:
            angles[i] = 0.0
        else:
            angles[i] = np.arccos(np.clip(np.dot(v_in, v_out) / (ni * no), -1.0, 1.0))
    return angles


def run_turning_point(seq_dir: Path, out_dir: Path, window: int = 10, smooth: int = 1) -> None:
    """Show frames around the sharpest EE direction reversal."""
    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]
    positions = ee_Ts[:, :3, 3]
    n_frames = len(positions)

    angles = _direction_change_angles(positions, smooth_k=smooth)
    turning_idx = int(np.argmax(angles))
    turning_deg = np.degrees(angles[turning_idx])
    print(f"Turning frame: {turning_idx}  ({turning_deg:.1f}°)")

    half = window // 2
    win_start = max(0, turning_idx - half)
    win_end   = min(n_frames, win_start + window)
    win_start = max(0, win_end - window)
    frame_indices = list(range(win_start, win_end))
    n_show = len(frame_indices)
    tp_col = turning_idx - win_start

    rs_path = seq_dir / "hdf5" / "realsense.h5"
    ev_path = seq_dir / "hdf5" / "events_cam0.h5"

    with h5py.File(rs_path, "r") as f:
        if win_start > 0:
            load_idx = [win_start - 1] + frame_indices
            all_rgb = f["rgb"][load_idx]
            pred_frame = all_rgb[0]; rgb_frames = all_rgb[1:]
        else:
            rgb_frames = f["rgb"][frame_indices]; pred_frame = None

    diff_frames = []
    for i in range(n_show):
        prev = pred_frame if (i == 0 and pred_frame is not None) else (rgb_frames[0] if i == 0 else rgb_frames[i - 1])
        diff_frames.append(rgb_frames[i].astype(np.int16) - prev.astype(np.int16))

    has_events = ev_path.exists()
    ev_frames_data = None
    if has_events:
        with h5py.File(ev_path, "r") as f:
            if "events/frames" in f:
                ev_frames_data = f["events/frames"][frame_indices]
            else:
                has_events = False

    n_rows = 2 + int(has_events)
    DIFF_ROW = 1 + int(has_events)
    fig, axes = plt.subplots(n_rows, n_show, figsize=(n_show * 2.0, n_rows * 2.2 + 1.2), squeeze=False)

    row_labels = ["RGB"]
    if has_events:
        row_labels.append("Events")
    row_labels.append("RGB diff")

    for col, fi in enumerate(frame_indices):
        is_tp = (col == tp_col)
        bc = "red" if is_tp else "none"

        ax = axes[0][col]
        ax.imshow(rgb_frames[col]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3 if is_tp else 0)
        title = f"#{fi}" + (f"\n↑ {turning_deg:.0f}°" if is_tp else "")
        ax.set_title(title, fontsize=7, color="red" if is_tp else "black", fontweight="bold" if is_tp else "normal")

        if has_events and ev_frames_data is not None:
            ax_ev = axes[1][col]
            ax_ev.imshow(normalize_for_display(ev_frames_data[col]), cmap="gray")
            ax_ev.set_xticks([]); ax_ev.set_yticks([])
            for sp in ax_ev.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3 if is_tp else 0)

        ax_d = axes[DIFF_ROW][col]
        diff_mag = np.abs(diff_frames[col]).mean(axis=2).astype(np.float32)
        ax_d.imshow(diff_mag, cmap="hot", vmin=0, vmax=diff_mag.max() or 1)
        ax_d.set_xticks([]); ax_d.set_yticks([])
        for sp in ax_d.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(3 if is_tp else 0)

        pos = ee_Ts[fi, :3, 3]
        ax_d.set_xlabel(f"x={pos[0]:.3f}\ny={pos[1]:.3f}\nz={pos[2]:.3f}", fontsize=5.5,
                        color="red" if is_tp else "black")

    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(label, fontsize=9)

    fig.suptitle(f"Turning point  |  {seq_dir.name}  |  frame {turning_idx} ({turning_deg:.1f}°)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "turning_point.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data analysis — generate image plots from recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Analysis to run")

    # -- poses --
    p = sub.add_parser("poses", help="Camera poses + RGB images")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--calib_dir", type=str, default="camera_data")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=None)

    # -- turning_point --
    p = sub.add_parser("turning_point", help="Frames around sharpest EE direction reversal")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--smooth", type=int, default=1)

    # -- all --
    p = sub.add_parser("all", help="Run both analyses")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--calib_dir", type=str, default="camera_data")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    seq_dir = Path(args.data_dir)
    out_dir = ensure_out_dir(seq_dir)

    cmd = args.command

    if cmd == "poses":
        run_poses(seq_dir, out_dir, calib_dir=args.calib_dir, n_samples=args.n, seed=args.seed)
    elif cmd == "turning_point":
        run_turning_point(seq_dir, out_dir, window=args.window, smooth=args.smooth)
    elif cmd == "all":
        print(f"Running all analyses for {seq_dir.name} → {out_dir}\n")
        errors = []
        for name, fn in [
            ("poses",         lambda: run_poses(seq_dir, out_dir, calib_dir=args.calib_dir)),
            ("turning_point", lambda: run_turning_point(seq_dir, out_dir)),
        ]:
            try:
                print(f"\n{'='*60}\n  {name}\n{'='*60}")
                fn()
            except Exception as e:
                print(f"  SKIPPED ({name}): {e}")
                errors.append(name)
        if errors:
            print(f"\nSkipped due to errors: {', '.join(errors)}")
        print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
