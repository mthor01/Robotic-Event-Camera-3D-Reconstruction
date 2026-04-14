#!/usr/bin/env python3
"""
Find the frame where the EE trajectory most abruptly reverses direction and
show the ±N RGB + event frames around it.

"Most abrupt direction change" = largest angle between consecutive velocity
vectors (i.e. the translation stops going one way and starts going another).

Usage:
    python3 visualize_turning_point.py --data_dir data/real/1
    python3 visualize_turning_point.py --data_dir data/real/1 --window 10 --out turning.png
    python3 visualize_turning_point.py --data_dir data/real/1 --smooth 5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_WINDOW = 10   # total frames to show (centred on the turning point)
DEFAULT_SMOOTH = 1    # velocity smoothing kernel size (1 = no smoothing)
DEFAULT_OUT    = str(Path(__file__).parent / "turning_point.png")


# ── helpers ────────────────────────────────────────────────────────────────────

def smooth(v: np.ndarray, k: int) -> np.ndarray:
    """Simple box-filter smoothing along axis 0; k=1 is a no-op."""
    if k <= 1:
        return v
    half = k // 2
    out = np.zeros_like(v)
    for i in range(len(v)):
        lo = max(0, i - half)
        hi = min(len(v), i + half + 1)
        out[i] = v[lo:hi].mean(axis=0)
    return out


def direction_change_angles(positions: np.ndarray, smooth_k: int = 1) -> np.ndarray:
    """
    Returns an array of length N where entry i is the angle (radians) between
    the velocity vector arriving at frame i and the one leaving frame i.

    Frames 0 and N-1 get 0 (no change computable at boundaries).

    positions: (N, 3) EE translation xyz.
    """
    pos = smooth(positions, smooth_k)
    # velocity[i] = pos[i+1] - pos[i], shape (N-1, 3)
    vel = np.diff(pos, axis=0)

    n = len(positions)
    angles = np.zeros(n)

    for i in range(1, n - 1):
        v_in  = vel[i - 1]
        v_out = vel[i]
        norm_in  = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        if norm_in < 1e-9 or norm_out < 1e-9:
            angles[i] = 0.0
        else:
            cos_a = np.clip(
                np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0
            )
            angles[i] = np.arccos(cos_a)

    return angles


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Scale any float/int array to uint8 [0, 255]."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize frames around the sharpest EE direction reversal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Recording dir (must contain hdf5/poses.h5, "
                             "hdf5/realsense.h5, hdf5/events_cam0.h5)")
    parser.add_argument("--window",  type=int, default=DEFAULT_WINDOW,
                        help="Total number of frames to show")
    parser.add_argument("--smooth",  type=int, default=DEFAULT_SMOOTH,
                        help="Velocity smoothing window (odd int, 1=none)")
    parser.add_argument("--out",     type=str, default=DEFAULT_OUT,
                        help="Output PNG path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    poses_path  = data_dir / "hdf5" / "poses.h5"
    rs_path     = data_dir / "hdf5" / "realsense.h5"
    ev_path     = data_dir / "hdf5" / "events_cam0.h5"

    # ── load poses ─────────────────────────────────────────────────────────────
    with h5py.File(poses_path, "r") as f:
        ee_Ts = f["ee_T"][:]          # (N, 4, 4)

    positions = ee_Ts[:, :3, 3]       # (N, 3) EE translation
    n_frames  = len(positions)

    # ── find turning point ─────────────────────────────────────────────────────
    angles = direction_change_angles(positions, smooth_k=args.smooth)
    turning_idx = int(np.argmax(angles))
    turning_angle_deg = np.degrees(angles[turning_idx])

    print(f"Frames in recording : {n_frames}")
    print(f"Turning point frame : {turning_idx}  "
          f"(direction change {turning_angle_deg:.1f}°)")

    # ── select window of frames ────────────────────────────────────────────────
    half = args.window // 2
    win_start = max(0, turning_idx - half)
    win_end   = min(n_frames, win_start + args.window)
    win_start = max(0, win_end - args.window)    # re-clamp if near end
    frame_indices = list(range(win_start, win_end))
    n_show = len(frame_indices)
    tp_col = turning_idx - win_start              # column index of turning point

    print(f"Showing frames      : {win_start} – {win_end - 1}  "
          f"(turning point at column {tp_col})")

    # ── load poses for the window ──────────────────────────────────────────────
    window_ee_Ts = ee_Ts[frame_indices]   # (n_show, 4, 4)

    # ── load image data ────────────────────────────────────────────────────────
    # Also fetch the predecessor of the first frame so we can compute a diff
    # for every column (including the first).
    with h5py.File(rs_path, "r") as f:
        if win_start > 0:
            load_indices = [win_start - 1] + frame_indices
            all_rgb = f["rgb"][load_indices]      # (n_show+1, H, W, 3) uint8
            pred_frame = all_rgb[0]
            rgb_frames = all_rgb[1:]
        else:
            rgb_frames = f["rgb"][frame_indices]  # (n_show, H, W, 3) uint8
            pred_frame = None

    # ── compute per-frame RGB diffs ────────────────────────────────────────────
    diff_frames = []
    for i in range(n_show):
        if i == 0:
            prev = pred_frame if pred_frame is not None else rgb_frames[0]
        else:
            prev = rgb_frames[i - 1]
        diff = rgb_frames[i].astype(np.int16) - prev.astype(np.int16)
        diff_frames.append(diff)

    has_events = ev_path.exists()
    if has_events:
        with h5py.File(ev_path, "r") as f:
            if "events/frames" in f:
                ev_frames = f["events/frames"][frame_indices]  # (n_show, H, W)
            else:
                has_events = False
                print("Warning: events/frames not found in events_cam0.h5 — "
                      "showing RGB only")

    # ── layout ─────────────────────────────────────────────────────────────────
    # Rows: RGB, (Events), RGB diff
    n_rows = 2 + int(has_events)   # RGB + diff always; events optional
    DIFF_ROW = 1 + int(has_events)  # index of the diff row
    fig_w  = n_show * 2.0
    fig_h  = n_rows * 2.2 + 1.2   # +1.2 for suptitle + pose text xlabels

    fig, axes = plt.subplots(
        n_rows, n_show,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    row_labels = ["RGB"]
    if has_events:
        row_labels.append("Events")
    row_labels.append("RGB diff\n(vs prev)")

    for col, frame_idx in enumerate(frame_indices):
        is_tp = (col == tp_col)
        border_color = "red" if is_tp else "none"

        # ── RGB row ────────────────────────────────────────────────────────────
        ax = axes[0][col]
        ax.imshow(rgb_frames[col])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_tp else 0)
        title = f"#{frame_idx}"
        if is_tp:
            title += f"\n↑ {turning_angle_deg:.0f}°"
        ax.set_title(title, fontsize=7,
                     color="red" if is_tp else "black",
                     fontweight="bold" if is_tp else "normal")

        # ── Events row ────────────────────────────────────────────────────────
        if has_events:
            ax_ev = axes[1][col]
            ev = normalize_for_display(ev_frames[col])
            ax_ev.imshow(ev, cmap="gray")
            ax_ev.set_xticks([]); ax_ev.set_yticks([])
            for spine in ax_ev.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3 if is_tp else 0)

        # ── RGB diff row ──────────────────────────────────────────────────────
        ax_diff = axes[DIFF_ROW][col]
        # Mean absolute change across RGB channels → magnitude image
        diff_mag = np.abs(diff_frames[col]).mean(axis=2).astype(np.float32)
        ax_diff.imshow(diff_mag, cmap="hot", vmin=0, vmax=diff_mag.max() or 1)
        ax_diff.set_xticks([]); ax_diff.set_yticks([])
        for spine in ax_diff.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3 if is_tp else 0)
        if col == 0 and pred_frame is None:
            ax_diff.set_title("(no prev)", fontsize=6, color="gray")

        # ── Pose annotation (EE xyz) under the diff row ───────────────────────
        pos = window_ee_Ts[col, :3, 3]   # (3,) metres
        pose_str = f"x={pos[0]:.3f}\ny={pos[1]:.3f}\nz={pos[2]:.3f}"
        ax_diff.set_xlabel(pose_str, fontsize=5.5, labelpad=2,
                           color="red" if is_tp else "black")

    # row labels on the left
    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(label, fontsize=9, labelpad=4)

    fig.suptitle(
        f"Frames around sharpest EE direction change  |  "
        f"recording: {data_dir.name}  |  "
        f"turning frame: {turning_idx}  ({turning_angle_deg:.1f}°)",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
