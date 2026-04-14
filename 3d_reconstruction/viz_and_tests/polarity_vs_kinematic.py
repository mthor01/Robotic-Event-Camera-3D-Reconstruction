#!/usr/bin/env python3
"""
Compare event-polarity change peaks against kinematic turning points.

For every drastic polarity-gradient peak (event-camera signal), find
the nearest kinematic turning point (translation local extremum or
rotation-velocity reversal in the EE poses), record the frame offset
between the two, and plot the distribution of those offsets.

Polarity peaks  – frames where |∇ smooth_signal| exceeds a percentile threshold.
Translation TPs – frames where any component of EE velocity changes sign
                  (local extremum in x, y, or z).
Rotation TPs    – frames where any component of the angular-velocity vector
                  changes sign (rotation reversal around that axis).

Output
------
  Single PNG with four panels:
    1. Polarity signal + detected peaks
    2. EE speed + detected translation turning points
    3. Angular speed + detected rotation turning points
    4. Histogram + rug of (polarity_peak − nearest_kinematic_TP) offsets

Usage
-----
    python3 polarity_vs_kinematic.py --data_dir data/real/1
    python3 polarity_vs_kinematic.py --data_dir data/real/1 --smooth 5
                                     --pol_pct 90 --kin_pct 80
    python3 polarity_vs_kinematic.py --data_dir data/real/1 --out offsets.png
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_SMOOTH   = 5     # polarity signal smoothing
DEFAULT_POL_PCT  = 85    # percentile threshold for polarity peaks
DEFAULT_KIN_PCT  = 70    # percentile threshold for kinematic turning-point magnitude
DEFAULT_MIN_DIST = 5     # minimum frames between detected peaks
DEFAULT_OUT      = str(Path(__file__).parent / "polarity_vs_kinematic.png")


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


def load_polarity_signal(seq_dir: Path) -> np.ndarray:
    for voxels_dir in [seq_dir / "events" / "voxels_cam0",
                       seq_dir / "events" / "voxels"]:
        npy_files = sorted(voxels_dir.glob("voxel_*.npy")) if voxels_dir.exists() else []
        if npy_files:
            print(f"Loading polarity  ({len(npy_files)} voxels) from {voxels_dir}")
            return np.array([np.load(f).mean() for f in npy_files], dtype=np.float32)
    ev_h5 = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5.exists():
        with h5py.File(ev_h5, "r") as f:
            if "events/frames" in f:
                frames = f["events/frames"][:]
                signal = frames.astype(np.float32).mean(axis=(1, 2)) - 128.0
                print(f"Loading polarity  ({len(signal)} frames) from events_cam0.h5")
                return signal
    raise FileNotFoundError(f"No event voxels or frames found in {seq_dir}.")


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """
    Extract the axis-angle rotation vector from a 3×3 rotation matrix.
    Returns a (3,) vector whose magnitude is the rotation angle [rad] and
    whose direction is the rotation axis (handled sign so angle >= 0).
    """
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def find_peaks_above_pct(signal: np.ndarray, pct: float,
                          min_dist: int) -> np.ndarray:
    """
    Return indices of local maxima in |signal| whose value exceeds the
    given percentile of |signal|.
    """
    mag = np.abs(signal)
    threshold = np.percentile(mag, pct)
    peaks, _ = find_peaks(mag, height=threshold, distance=min_dist)
    return peaks


def sign_change_indices(v: np.ndarray) -> np.ndarray:
    """
    Return indices i where v[i-1] and v[i] have opposite signs
    (zero-crossings of a velocity / angular-velocity component).
    """
    signs = np.sign(v)
    # ignore exact zeros
    for i in range(len(signs)):
        if signs[i] == 0 and i > 0:
            signs[i] = signs[i - 1]
    crossings = np.where(np.diff(signs) != 0)[0] + 1
    return crossings


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Offset between polarity peaks and kinematic turning points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",  type=str, required=True)
    parser.add_argument("--smooth",    type=int, default=DEFAULT_SMOOTH,
                        help="Polarity signal smoothing window")
    parser.add_argument("--pol_pct",   type=float, default=DEFAULT_POL_PCT,
                        help="Percentile threshold for polarity-gradient peaks")
    parser.add_argument("--kin_pct",   type=float, default=DEFAULT_KIN_PCT,
                        help="Percentile threshold for kinematic-event magnitude")
    parser.add_argument("--min_dist",  type=int, default=DEFAULT_MIN_DIST,
                        help="Minimum frames between detected peaks")
    parser.add_argument("--out",       type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)

    # ── polarity signal ────────────────────────────────────────────────────────
    raw_signal   = load_polarity_signal(seq_dir)
    n_frames     = len(raw_signal)
    smooth_sig   = box_smooth(raw_signal, args.smooth)
    pol_grad     = np.gradient(smooth_sig.astype(np.float64))

    pol_peaks = find_peaks_above_pct(pol_grad, args.pol_pct, args.min_dist)
    print(f"Polarity peaks found : {len(pol_peaks)} "
          f"(|∇signal| > {args.pol_pct}th pct, min_dist={args.min_dist})")

    # ── EE poses ───────────────────────────────────────────────────────────────
    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]           # (N, 4, 4)

    positions = ee_Ts[:, :3, 3]        # (N, 3)
    rotations = ee_Ts[:, :3, :3]       # (N, 3, 3)

    # ── translation turning points ─────────────────────────────────────────────
    # Velocity = finite diff of position.  TP = sign change in any component.
    vel = np.diff(positions, axis=0)           # (N-1, 3)
    # Pad to length N (no TP at index 0 or N-1)
    trans_tp_sets = []
    for axis in range(3):
        crossings = sign_change_indices(vel[:, axis])   # indices into vel → frame idx
        trans_tp_sets.append(set(crossings.tolist()))
    all_trans_tps = sorted(set.union(*trans_tp_sets))

    # Keep only those whose speed magnitude exceeds threshold
    speed = np.linalg.norm(vel, axis=1)             # (N-1,)
    # Evaluate speed at each translational TP
    if all_trans_tps:
        tp_speeds = np.array([speed[i] if i < len(speed) else 0
                              for i in all_trans_tps])
        spd_thresh = np.percentile(speed, args.kin_pct)
        all_trans_tps = [i for i, s in zip(all_trans_tps, tp_speeds)
                         if s >= spd_thresh]

    print(f"Translation TPs found: {len(all_trans_tps)}")

    # ── rotation turning points ────────────────────────────────────────────────
    # Angular velocity vector = rotation_vector(R[i-1].T @ R[i])
    ang_vel = np.zeros((n_frames - 1, 3))
    for i in range(1, n_frames):
        R_rel = rotations[i - 1].T @ rotations[i]
        ang_vel[i - 1] = rotation_vector(R_rel)

    rot_tp_sets = []
    for axis in range(3):
        crossings = sign_change_indices(ang_vel[:, axis])
        rot_tp_sets.append(set(crossings.tolist()))
    all_rot_tps = sorted(set.union(*rot_tp_sets))

    ang_speed = np.linalg.norm(ang_vel, axis=1)
    if all_rot_tps:
        tp_ang_speeds = np.array([ang_speed[i] if i < len(ang_speed) else 0
                                  for i in all_rot_tps])
        ang_thresh = np.percentile(ang_speed, args.kin_pct)
        all_rot_tps = [i for i, s in zip(all_rot_tps, tp_ang_speeds)
                       if s >= ang_thresh]

    print(f"Rotation TPs found   : {len(all_rot_tps)}")

    # ── combine kinematic TPs ──────────────────────────────────────────────────
    all_kin_tps = np.array(sorted(set(all_trans_tps) | set(all_rot_tps)))
    print(f"Combined kin TPs     : {len(all_kin_tps)}")

    # ── compute offsets ────────────────────────────────────────────────────────
    # For each polarity peak, find the nearest kinematic TP.
    offsets = []
    matched_kin = []
    if len(all_kin_tps) > 0:
        for pp in pol_peaks:
            dists = np.abs(all_kin_tps - pp)
            nearest_idx = int(np.argmin(dists))
            nearest_kin = int(all_kin_tps[nearest_idx])
            offset = int(pp) - nearest_kin
            offsets.append(offset)
            matched_kin.append(nearest_kin)
            print(f"  polarity peak {pp:4d}  →  nearest kin TP {nearest_kin:4d}  "
                  f"(offset {offset:+d})")
    else:
        print("No kinematic TPs found — try lowering --kin_pct")

    offsets = np.array(offsets, dtype=int)

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.50)
    x = np.arange(n_frames)

    # Row 1 – polarity signal + peaks
    ax = axes[0]
    ax.plot(x, raw_signal, color="steelblue", lw=0.7, alpha=0.4, label="raw")
    ax.plot(x, smooth_sig, color="steelblue", lw=1.4, label=f"smoothed (k={args.smooth})")
    ax2r = ax.twinx()
    ax2r.plot(x, pol_grad, color="orange", lw=1.0, alpha=0.7, label="|∇signal|")
    ax2r.set_ylabel("|∇ signal|", fontsize=8, color="orange")
    ax2r.tick_params(axis="y", labelcolor="orange", labelsize=7)
    for pp in pol_peaks:
        ax.axvline(pp, color="red", lw=0.8, alpha=0.6)
    ax.set_ylabel("Net polarity")
    ax.set_title(f"Polarity signal  ({len(pol_peaks)} peaks >  {args.pol_pct}th pct)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    # Row 2 – EE speed + translation TPs
    ax = axes[1]
    speed_full = np.concatenate([speed, [0]])   # pad to N
    ax.plot(x, speed_full * 1000, color="darkgreen", lw=1.2, label="EE speed [mm/frame]")
    for tp in all_trans_tps:
        ax.axvline(tp, color="purple", lw=0.8, alpha=0.5)
    ax.set_ylabel("EE speed [mm/frame]")
    ax.set_title(f"Translation turning points  ({len(all_trans_tps)} above {args.kin_pct}th pct speed)", fontsize=9)
    ax.legend(fontsize=7)

    # Row 3 – angular speed + rotation TPs
    ax = axes[2]
    ang_full = np.concatenate([np.degrees(ang_speed), [0]])
    ax.plot(x, ang_full, color="sienna", lw=1.2, label="Angular speed [°/frame]")
    for tp in all_rot_tps:
        ax.axvline(tp, color="teal", lw=0.8, alpha=0.5)
    ax.set_ylabel("Angular speed [°/frame]")
    ax.set_title(f"Rotation turning points  ({len(all_rot_tps)} above {args.kin_pct}th pct ang speed)", fontsize=9)
    ax.legend(fontsize=7)

    # Row 4 – offset histogram + rug
    ax = axes[3]
    if len(offsets) > 0:
        max_abs = max(abs(offsets.min()), abs(offsets.max()), 1)
        bins = np.arange(-max_abs - 1, max_abs + 2) - 0.5
        ax.hist(offsets, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        # rug plot
        ax.plot(offsets, np.zeros_like(offsets) - 0.05, "|", color="steelblue",
                ms=12, markeredgewidth=1.5, clip_on=False)
        ax.axvline(0, color="red", lw=1.5, ls="--", label="exact match")
        ax.axvline(np.median(offsets), color="orange", lw=1.5, ls=":",
                   label=f"median={np.median(offsets):.1f}")
        ax.set_xlabel("Polarity peak frame  −  nearest kinematic TP frame")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Offset distribution  |  n={len(offsets)}  "
            f"mean={offsets.mean():.1f}  std={offsets.std():.1f}  "
            f"median={np.median(offsets):.1f}",
            fontsize=9,
        )
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No offsets computed", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Offset distribution", fontsize=9)

    fig.suptitle(
        f"Polarity peaks vs kinematic turning points  |  {seq_dir.name}",
        fontsize=11,
    )

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
