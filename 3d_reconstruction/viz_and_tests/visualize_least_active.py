#!/usr/bin/env python3
"""
Find the frame with the least event activity (minimum total absolute voxel
energy) and plot how translation speed, angular speed, and event activity
evolve in a window around that frame.

Event activity = sum of |voxel| values per frame (total energy, not net polarity).

Usage:
    python3 visualize_least_active.py --data_dir data/real/1
    python3 visualize_least_active.py --data_dir data/real/1 --window 40 --smooth 3
    python3 visualize_least_active.py --data_dir data/real/1 --out quiet.png
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_WINDOW = 40
DEFAULT_SMOOTH = 3
DEFAULT_OUT    = str(Path(__file__).parent / "least_active.png")


# ── helpers ────────────────────────────────────────────────────────────────────

def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.copy()
    half = k // 2
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def load_activity_and_polarity(seq_dir: Path):
    """
    Returns (activity, polarity) arrays of length N_frames.
    activity  = mean(|voxel|)  per frame  – total event energy
    polarity  = mean(voxel)    per frame  – net signed polarity
    """
    for vdir in [seq_dir / "events" / "voxels_cam0",
                 seq_dir / "events" / "voxels"]:
        files = sorted(vdir.glob("voxel_*.npy")) if vdir.exists() else []
        if files:
            print(f"Loading {len(files)} voxels from {vdir}")
            activity = np.array([np.abs(np.load(f)).mean() for f in files],
                                dtype=np.float32)
            polarity = np.array([np.load(f).mean() for f in files],
                                dtype=np.float32)
            return activity, polarity

    ev_h5 = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5.exists():
        with h5py.File(ev_h5, "r") as f:
            if "events/frames" in f:
                frames = f["events/frames"][:].astype(np.float32)
                centred = frames - 128.0
                activity = np.abs(centred).mean(axis=(1, 2))
                polarity = centred.mean(axis=(1, 2))
                print(f"Loading {len(activity)} frames from events_cam0.h5")
                return activity, polarity

    raise FileNotFoundError(f"No event data found in {seq_dir}")


def rotation_angle_per_frame(rotations: np.ndarray) -> np.ndarray:
    n = len(rotations)
    ang = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        R_rel = rotations[i - 1].T @ rotations[i]
        cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        ang[i] = float(np.arccos(cos_a))
    return ang


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Visualize the quietest (least-event) moment in a recording.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--window",   type=int, default=DEFAULT_WINDOW,
                        help="Number of frames to show around the quiet point")
    parser.add_argument("--smooth",   type=int, default=DEFAULT_SMOOTH,
                        help="Box-filter width for display signals")
    parser.add_argument("--out",      type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)

    # ── signals ────────────────────────────────────────────────────────────────
    activity, polarity = load_activity_and_polarity(seq_dir)
    n = len(activity)

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]

    positions = ee_Ts[:, :3, 3]
    rotations = ee_Ts[:, :3, :3]

    trans_speed = np.zeros(n, dtype=np.float32)
    trans_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 1000  # mm

    ang_speed = rotation_angle_per_frame(rotations)   # rad/frame
    ang_speed_deg = np.degrees(ang_speed)

    # ── find quietest frame ────────────────────────────────────────────────────
    quiet_idx = int(np.argmin(box_smooth(activity, args.smooth)))
    quiet_activity = activity[quiet_idx]

    print(f"Frames in recording  : {n}")
    print(f"Quietest frame       : {quiet_idx}  "
          f"(activity = {quiet_activity:.5f})")
    print(f"Translation speed    : {trans_speed[quiet_idx]:.3f} mm/frame")
    print(f"Angular speed        : {ang_speed_deg[quiet_idx]:.4f} °/frame")

    # ── window ─────────────────────────────────────────────────────────────────
    half = args.window // 2
    win_start = max(0, quiet_idx - half)
    win_end   = min(n, win_start + args.window)
    win_start = max(0, win_end - args.window)
    win_x     = np.arange(win_start, win_end)
    off_x     = win_x - quiet_idx           # signed offsets

    act_win   = activity[win_start:win_end]
    pol_win   = polarity[win_start:win_end]
    trans_win = trans_speed[win_start:win_end]
    ang_win   = ang_speed_deg[win_start:win_end]

    # smooth window signals for display
    act_s   = box_smooth(act_win,   args.smooth)
    pol_s   = box_smooth(pol_win,   args.smooth)
    trans_s = box_smooth(trans_win, args.smooth)
    ang_s   = box_smooth(ang_win,   args.smooth)

    # ── figure: 3 rows ─────────────────────────────────────────────────────────
    #   Row 1 – full activity signal, quiet point marked
    #   Row 2 – window: event activity + net polarity
    #   Row 3 – window: translation speed + angular speed
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.45)

    # ── Row 1: full activity signal ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(np.arange(n), activity, color="steelblue", lw=0.8, alpha=0.5, label="raw activity")
    ax.plot(np.arange(n), box_smooth(activity, args.smooth),
            color="steelblue", lw=1.5, label=f"smoothed (k={args.smooth})")
    ax.axvspan(win_start, win_end - 1, alpha=0.12, color="orange", label="window")
    ax.axvline(quiet_idx, color="red", lw=1.5, ls="--",
               label=f"quietest frame #{quiet_idx}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Event activity\n(mean |voxel|)")
    ax.set_title("Event activity over full recording", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # ── Row 2: window – activity + polarity ────────────────────────────────────
    ax = axes[1]
    ax.plot(off_x, act_s,  color="steelblue",  lw=1.5, label="event activity")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7, label=f"quiet frame #{quiet_idx}")
    ax.set_ylabel("Event activity", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")

    ax_r = ax.twinx()
    ax_r.plot(off_x, pol_s, color="darkorange", lw=1.5, ls="--", label="net polarity")
    ax_r.axhline(0, color="gray", lw=0.6, ls=":")
    ax_r.set_ylabel("Net polarity (mean voxel)", color="darkorange")
    ax_r.tick_params(axis="y", labelcolor="darkorange")

    ax.set_xlabel("Frame offset from quiet point")
    ax.set_title("Event activity & net polarity around quiet point", fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    # ── Row 3: window – translation + angular speed ────────────────────────────
    ax = axes[2]
    ax.plot(off_x, trans_s, color="forestgreen", lw=1.5, label="translation speed [mm/frame]")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7)
    ax.set_ylabel("Translation speed [mm/frame]", color="forestgreen")
    ax.tick_params(axis="y", labelcolor="forestgreen")

    ax_r2 = ax.twinx()
    ax_r2.plot(off_x, ang_s, color="tomato", lw=1.5, ls="--", label="angular speed [°/frame]")
    ax_r2.set_ylabel("Angular speed [°/frame]", color="tomato")
    ax_r2.tick_params(axis="y", labelcolor="tomato")

    ax.set_xlabel("Frame offset from quiet point")
    ax.set_title("Translation & angular speed around quiet point", fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    fig.suptitle(
        f"Least-active event frame  |  {seq_dir.name}  |  "
        f"quiet frame: {quiet_idx}  (activity={quiet_activity:.5f})",
        fontsize=10,
    )

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
