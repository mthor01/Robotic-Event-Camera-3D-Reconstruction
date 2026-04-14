#!/usr/bin/env python3
"""
Detect the frame with the sharpest event-polarity reversal (indicating a
robot direction change) and plot relative pose changes from that frame to
its surrounding frames.

Detection method
----------------
Voxel-grid bins encode signed event counts (+polarity ≈ +1, -polarity ≈ -1).
The per-frame net polarity signal  s[t] = mean(voxel[t])  tracks the dominant
direction of motion.  A direction reversal → sign flip / largest gradient in s.

Output figure
-------------
  Row 1  – full polarity signal over the recording, with the detection window
            highlighted and the turning point marked.
  Row 2  – relative translation (dx, dy, dz) from the turning frame to every
            frame in the window.
  Row 3  – relative rotation angle (degrees) from the turning frame to every
            frame in the window.

Usage
-----
    python3 visualize_direction_change.py --data_dir data/real/1
    python3 visualize_direction_change.py --data_dir data/real/1 --window 20
    python3 visualize_direction_change.py --data_dir data/real/1 --smooth 5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_WINDOW = 20
DEFAULT_SMOOTH = 3
DEFAULT_OUT    = str(Path(__file__).parent / "direction_change.png")


# ── helpers ────────────────────────────────────────────────────────────────────

def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    """1-D box-filter along axis 0; k=1 is a no-op."""
    if k <= 1:
        return arr.copy()
    half = k // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def load_polarity_signal(seq_dir: Path) -> np.ndarray:
    """
    Return per-frame net-polarity signal (1-D float array, length = N_frames).

    Tries (in order):
      1. events/voxels_cam0/voxel_NNNNNN.npy  – precomputed voxels
      2. events/voxels/voxel_NNNNNN.npy
      3. hdf5/events_cam0.h5  events/frames   – fallback: mean pixel value
    """
    for voxels_dir in [seq_dir / "events" / "voxels_cam0",
                       seq_dir / "events" / "voxels"]:
        npy_files = sorted(voxels_dir.glob("voxel_*.npy")) if voxels_dir.exists() else []
        if npy_files:
            print(f"Loading polarity from {voxels_dir}  ({len(npy_files)} voxels)")
            signal = np.array([np.load(f).mean() for f in npy_files], dtype=np.float32)
            return signal

    # Fallback: events_cam0.h5 frames
    ev_h5 = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5.exists():
        with h5py.File(ev_h5, "r") as f:
            if "events/frames" in f:
                frames = f["events/frames"][:]          # (N, H, W) uint8
                # Assume gray=128 means zero net polarity
                signal = frames.astype(np.float32).mean(axis=(1, 2)) - 128.0
                print(f"Loading polarity from events/frames fallback  ({len(signal)} frames)")
                return signal
    raise FileNotFoundError(
        f"No event voxels or frames found in {seq_dir}. "
        "Run precompute_voxels.py first."
    )


def find_polarity_reversal(signal: np.ndarray, smooth_k: int) -> int:
    """
    Return the index of the sharpest polarity change.

    We look for the largest magnitude of the first derivative of the smoothed
    signal — i.e. the frame where the net polarity changes fastest.
    """
    s = box_smooth(signal, smooth_k)
    grad = np.gradient(s)
    return int(np.argmax(np.abs(grad)))


def relative_pose(T_ref: np.ndarray, T_other: np.ndarray):
    """
    Compute the relative SE(3) transform from T_ref to T_other.
    Returns (dt, angle_deg) where
      dt        : (3,) translation delta in reference frame [metres]
      angle_deg : scalar rotation angle [degrees]
    """
    T_rel = np.linalg.inv(T_ref) @ T_other
    dt = T_rel[:3, 3]
    R  = T_rel[:3, :3]
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    return dt, angle_deg


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect event-polarity reversal and plot relative pose changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Recording dir (hdf5/poses.h5 + events/voxels_cam0/)")
    parser.add_argument("--window",  type=int, default=DEFAULT_WINDOW,
                        help="Number of frames to show around the turning point")
    parser.add_argument("--smooth",  type=int, default=DEFAULT_SMOOTH,
                        help="Box-filter width for polarity smoothing (1=none)")
    parser.add_argument("--out",     type=str, default=DEFAULT_OUT,
                        help="Output PNG path")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)

    # ── load polarity signal ───────────────────────────────────────────────────
    signal = load_polarity_signal(seq_dir)
    n_frames = len(signal)

    # ── detect turning point ───────────────────────────────────────────────────
    turning_idx = find_polarity_reversal(signal, smooth_k=args.smooth)
    smooth_signal = box_smooth(signal, args.smooth)

    print(f"Frames in recording : {n_frames}")
    print(f"Turning point frame : {turning_idx}  "
          f"(polarity gradient {np.gradient(smooth_signal)[turning_idx]:.4f})")

    # ── select window ──────────────────────────────────────────────────────────
    half = args.window // 2
    win_start = max(0, turning_idx - half)
    win_end   = min(n_frames, win_start + args.window)
    win_start = max(0, win_end - args.window)
    frame_indices = list(range(win_start, win_end))
    n_show = len(frame_indices)
    tp_col = turning_idx - win_start

    print(f"Window frames       : {win_start} – {win_end - 1}")

    # ── load EE poses ──────────────────────────────────────────────────────────
    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]                      # (N, 4, 4)

    T_turning = ee_Ts[turning_idx]               # reference pose

    # ── compute relative poses ─────────────────────────────────────────────────
    offsets = [i - turning_idx for i in frame_indices]   # signed frame offsets
    dxs, dys, dzs, d_rots = [], [], [], []
    # Facing direction: +Z column of EE rotation, expressed in the turning
    # frame's orientation.  Components stay in [-1, 1] (unit vector).
    # At offset 0 this equals [0, 0, 1] by definition.
    fxs, fys, fzs = [], [], []

    R_ref  = T_turning[:3, :3]
    Z_AXIS = np.array([0.0, 0.0, 1.0])

    for idx in frame_indices:
        dt, angle = relative_pose(T_turning, ee_Ts[idx])
        dxs.append(dt[0] * 1000)   # mm for readability
        dys.append(dt[1] * 1000)
        dzs.append(dt[2] * 1000)
        d_rots.append(angle)
        R_i    = ee_Ts[idx, :3, :3]
        R_rel  = R_ref.T @ R_i          # relative rotation matrix
        facing = R_rel @ Z_AXIS         # where +Z now points in turning-frame coords
        fxs.append(facing[0])
        fys.append(facing[1])
        fzs.append(facing[2])

    dxs    = np.array(dxs)
    dys    = np.array(dys)
    dzs    = np.array(dzs)
    d_rots = np.array(d_rots)
    fxs    = np.array(fxs)
    fys    = np.array(fys)
    fzs    = np.array(fzs)

    # ── console summary ────────────────────────────────────────────────────────
    print(f"\n{'Offset':>8}  {'frame':>6}  {'dx[mm]':>9}  "
          f"{'dy[mm]':>9}  {'dz[mm]':>9}  {'|drot|[°]':>10}")
    print("-" * 60)
    for k, (off, idx) in enumerate(zip(offsets, frame_indices)):
        marker = " ← turning" if off == 0 else ""
        print(f"{off:+8d}  {idx:6d}  {dxs[k]:+9.2f}  "
              f"{dys[k]:+9.2f}  {dzs[k]:+9.2f}  {d_rots[k]:10.2f}{marker}")

    # ── figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(10, n_show * 0.7), 12))
    gs  = fig.add_gridspec(4, 1, hspace=0.50)

    x_all = np.arange(n_frames)

    # ── Row 1: full polarity signal ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_all, signal, color="steelblue", lw=0.8, alpha=0.5, label="raw")
    ax1.plot(x_all, smooth_signal, color="steelblue", lw=1.5, label=f"smoothed (k={args.smooth})")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    # highlight window
    ax1.axvspan(win_start, win_end - 1, alpha=0.12, color="orange", label="window")
    ax1.axvline(turning_idx, color="red", lw=1.5, ls="--", label=f"turning (#{turning_idx})")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Net polarity\n(mean voxel value)")
    ax1.set_title("Event polarity signal over recording", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")

    # ── Row 2: relative translation ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(offsets, dxs, "r.-", lw=1.4, ms=5, label="dx")
    ax2.plot(offsets, dys, "g.-", lw=1.4, ms=5, label="dy")
    ax2.plot(offsets, dzs, "b.-", lw=1.4, ms=5, label="dz")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvline(0, color="red",  lw=1.5, ls="--", alpha=0.6)
    ax2.set_xlabel("Frame offset from turning point")
    ax2.set_ylabel("Δ translation [mm]")
    ax2.set_title("Relative translation from turning frame", fontsize=10)
    ax2.legend(fontsize=8, loc="best")
    ax2.set_xticks(offsets)
    ax2.set_xticklabels([str(o) for o in offsets], fontsize=7)

    # ── Row 3: relative facing direction ──────────────────────────────────────
    # fx, fy, fz are the components of the EE +Z axis expressed in the turning
    # frame's orientation (unit vector, range [-1, 1]).  At offset 0 = [0,0,1].
    # A sign flip in any component shows a genuine rotation reversal around
    # that axis.
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(offsets, fxs, "r.-", lw=1.4, ms=5, label="facing x")
    ax3.plot(offsets, fys, "g.-", lw=1.4, ms=5, label="facing y")
    ax3.plot(offsets, fzs, "b.-", lw=1.4, ms=5, label="facing z")
    ax3.axhline(0, color="gray", lw=0.8, ls="--")
    ax3.axvline(0, color="red",  lw=1.5, ls="--", alpha=0.6)
    ax3.set_ylim(-1.05, 1.05)
    ax3.set_xlabel("Frame offset from turning point")
    ax3.set_ylabel("Facing direction\n(unit vector component)")
    ax3.set_title(
        "Relative facing direction (+Z of EE, in turning-frame coords)\n"
        "sign flip = rotation reversal around that axis",
        fontsize=10,
    )
    ax3.legend(fontsize=8, loc="best")
    ax3.set_xticks(offsets)
    ax3.set_xticklabels([str(o) for o in offsets], fontsize=7)

    # ── Row 4: geodesic rotation angle ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(offsets, d_rots, "k.-", lw=1.4, ms=5)
    ax4.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6)
    ax4.set_xlabel("Frame offset from turning point")
    ax4.set_ylabel("Δ rotation [°]")
    ax4.set_title("Geodesic rotation distance from turning frame", fontsize=10)
    ax4.set_xticks(offsets)
    ax4.set_xticklabels([str(o) for o in offsets], fontsize=7)
    ax4.set_ylim(bottom=0)

    fig.suptitle(
        f"Direction change detection via event polarity  |  "
        f"{seq_dir.name}  |  turning frame: {turning_idx}",
        fontsize=10,
    )

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
