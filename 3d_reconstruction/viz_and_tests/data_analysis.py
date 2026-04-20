#!/usr/bin/env python3
"""
Unified data analysis script — generates image plots from recording data.

Merges the following individual scripts into subcommands:
  polarity_speed     – Overlay net polarity, translation speed, angular speed
  polarity_kinematic – Polarity peaks vs kinematic turning points
  sample             – Random input / depth-GT pair from dataset
  direction_change   – Detect polarity reversal & plot relative pose changes
  least_active       – Find and visualise the quietest event frame
  poses              – Camera poses with RGB images (3-D + grid)
  raw_poses          – Raw pose data (position, orientation, joints, rate)
  turning_point      – Frames around sharpest EE direction reversal
  voxels             – Voxel grids with event / RGB frames
  white_mask         – White-pixel mask projection onto event plane
  all                – Run every analysis above

Output images are saved to  <data_dir>/data_plots/  by default.

Usage examples:
    python data_analysis.py polarity_speed   --data_dir data/real/1
    python data_analysis.py direction_change --data_dir data/real/1 --smooth 5
    python data_analysis.py all              --data_dir data/real/1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks


# ══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════

def ensure_out_dir(data_dir: Path) -> Path:
    """Return <data_dir>/data_plots, creating it if necessary."""
    d = data_dir / "data_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    """1-D box filter of width *k*.  k <= 1 is a no-op."""
    if k <= 1:
        return arr.astype(np.float64).copy()
    half = k // 2
    out = np.empty(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].astype(np.float64).mean()
    return out


def zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    return (arr - arr.mean()) / std if std > 0 else arr - arr.mean()


def load_polarity_signal(seq_dir: Path) -> np.ndarray:
    """Per-frame net polarity from voxels or event frames."""
    for vdir in [seq_dir / "events" / "voxels_cam0",
                 seq_dir / "events" / "voxels"]:
        files = sorted(vdir.glob("voxel_*.npy")) if vdir.exists() else []
        if files:
            return np.array([np.load(f).mean() for f in files], dtype=np.float32)
    ev_h5 = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5.exists():
        with h5py.File(ev_h5, "r") as f:
            if "events/frames" in f:
                frames = f["events/frames"][:]
                return frames.astype(np.float32).mean(axis=(1, 2)) - 128.0
    raise FileNotFoundError(f"No event data found in {seq_dir}")


def load_activity_and_polarity(seq_dir: Path):
    """Return (activity, polarity) arrays of length N_frames."""
    for vdir in [seq_dir / "events" / "voxels_cam0",
                 seq_dir / "events" / "voxels"]:
        files = sorted(vdir.glob("voxel_*.npy")) if vdir.exists() else []
        if files:
            activity = np.array([np.abs(np.load(f)).mean() for f in files], dtype=np.float32)
            polarity = np.array([np.load(f).mean() for f in files], dtype=np.float32)
            return activity, polarity
    ev_h5 = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5.exists():
        with h5py.File(ev_h5, "r") as f:
            if "events/frames" in f:
                frames = f["events/frames"][:].astype(np.float32)
                centred = frames - 128.0
                return np.abs(centred).mean(axis=(1, 2)), centred.mean(axis=(1, 2))
    raise FileNotFoundError(f"No event data found in {seq_dir}")


def rotation_angle_per_frame(rotations: np.ndarray) -> np.ndarray:
    """Angular step (rad) between consecutive rotation matrices."""
    n = len(rotations)
    ang = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        R_rel = rotations[i - 1].T @ rotations[i]
        cos_a = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        ang[i] = float(np.arccos(cos_a))
    return ang


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector from a 3×3 rotation matrix."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def rotation_matrix_to_euler_zyx(R: np.ndarray):
    """ZYX Euler angles (yaw, pitch, roll) in degrees from (N,3,3) rotation matrices."""
    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6
    roll  = np.where(singular, np.arctan2(-R[:, 1, 2], R[:, 1, 1]),
                               np.arctan2( R[:, 2, 1], R[:, 2, 2]))
    pitch = np.where(singular, np.arctan2(-R[:, 2, 0], sy),
                               np.arctan2(-R[:, 2, 0], sy))
    yaw   = np.where(singular, np.zeros(len(R)),
                               np.arctan2( R[:, 1, 0], R[:, 0, 0]))
    return np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)


def find_peaks_above_pct(signal: np.ndarray, pct: float, min_dist: int) -> np.ndarray:
    mag = np.abs(signal)
    threshold = np.percentile(mag, pct)
    peaks, _ = find_peaks(mag, height=threshold, distance=min_dist)
    return peaks


def sign_change_indices(v: np.ndarray) -> np.ndarray:
    signs = np.sign(v)
    for i in range(len(signs)):
        if signs[i] == 0 and i > 0:
            signs[i] = signs[i - 1]
    return np.where(np.diff(signs) != 0)[0] + 1


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
#  1. polarity_speed
# ══════════════════════════════════════════════════════════════════════

def run_polarity_speed(seq_dir: Path, out_dir: Path, smooth: int = 5) -> None:
    """Overlay net event polarity, EE translation speed, and angular speed."""
    polarity = load_polarity_signal(seq_dir)
    n = len(polarity)
    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]
    positions = ee_Ts[:, :3, 3]
    rotations = ee_Ts[:, :3, :3]

    trans_speed = np.zeros(n, dtype=np.float32)
    trans_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    ang_speed = rotation_angle_per_frame(rotations)

    pol_s   = zscore(box_smooth(polarity, smooth))
    trans_s = zscore(box_smooth(trans_speed, smooth))
    ang_s   = zscore(box_smooth(np.degrees(ang_speed), smooth))

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(n)
    ax.plot(x, pol_s,   color="steelblue",   lw=1.2, alpha=0.9,  label="net polarity")
    ax.plot(x, trans_s, color="forestgreen",  lw=1.2, alpha=0.85, label="translation speed")
    ax.plot(x, ang_s,   color="tomato",       lw=1.2, alpha=0.85, label="angular speed")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Z-score")
    ax.set_title(f"Polarity · translation · angular speed  |  {seq_dir.name}  (k={smooth})", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out = out_dir / "polarity_vs_speed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  2. polarity_kinematic
# ══════════════════════════════════════════════════════════════════════

def run_polarity_kinematic(
    seq_dir: Path, out_dir: Path,
    smooth: int = 5, pol_pct: float = 85, kin_pct: float = 70, min_dist: int = 5,
) -> None:
    """Compare polarity-gradient peaks against kinematic turning points."""
    raw_signal = load_polarity_signal(seq_dir)
    n_frames = len(raw_signal)
    smooth_sig = box_smooth(raw_signal, smooth).astype(np.float32)
    pol_grad = np.gradient(smooth_sig.astype(np.float64))
    pol_peaks = find_peaks_above_pct(pol_grad, pol_pct, min_dist)
    print(f"Polarity peaks: {len(pol_peaks)}")

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]
    positions = ee_Ts[:, :3, 3]
    rotations = ee_Ts[:, :3, :3]

    # Translation turning points
    vel = np.diff(positions, axis=0)
    trans_tp_sets = [set(sign_change_indices(vel[:, ax]).tolist()) for ax in range(3)]
    all_trans_tps = sorted(set.union(*trans_tp_sets)) if trans_tp_sets else []
    speed = np.linalg.norm(vel, axis=1)
    if all_trans_tps:
        tp_speeds = np.array([speed[i] if i < len(speed) else 0 for i in all_trans_tps])
        spd_thresh = np.percentile(speed, kin_pct)
        all_trans_tps = [i for i, s in zip(all_trans_tps, tp_speeds) if s >= spd_thresh]

    # Rotation turning points
    ang_vel = np.zeros((n_frames - 1, 3))
    for i in range(1, n_frames):
        ang_vel[i - 1] = rotation_vector(rotations[i - 1].T @ rotations[i])
    rot_tp_sets = [set(sign_change_indices(ang_vel[:, ax]).tolist()) for ax in range(3)]
    all_rot_tps = sorted(set.union(*rot_tp_sets)) if rot_tp_sets else []
    ang_speed = np.linalg.norm(ang_vel, axis=1)
    if all_rot_tps:
        tp_ang = np.array([ang_speed[i] if i < len(ang_speed) else 0 for i in all_rot_tps])
        ang_thresh = np.percentile(ang_speed, kin_pct)
        all_rot_tps = [i for i, s in zip(all_rot_tps, tp_ang) if s >= ang_thresh]

    all_kin_tps = np.array(sorted(set(all_trans_tps) | set(all_rot_tps)))
    print(f"Translation TPs: {len(all_trans_tps)}, Rotation TPs: {len(all_rot_tps)}, Combined: {len(all_kin_tps)}")

    offsets, matched_kin = [], []
    if len(all_kin_tps) > 0:
        for pp in pol_peaks:
            nearest_idx = int(np.argmin(np.abs(all_kin_tps - pp)))
            nearest_kin = int(all_kin_tps[nearest_idx])
            offsets.append(int(pp) - nearest_kin)
            matched_kin.append(nearest_kin)
    offsets = np.array(offsets, dtype=int)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.50)
    x = np.arange(n_frames)

    ax = axes[0]
    ax.plot(x, raw_signal, color="steelblue", lw=0.7, alpha=0.4, label="raw")
    ax.plot(x, smooth_sig, color="steelblue", lw=1.4, label=f"smoothed (k={smooth})")
    ax2r = ax.twinx()
    ax2r.plot(x, pol_grad, color="orange", lw=1.0, alpha=0.7, label="|∇signal|")
    ax2r.set_ylabel("|∇ signal|", fontsize=8, color="orange")
    ax2r.tick_params(axis="y", labelcolor="orange", labelsize=7)
    for pp in pol_peaks:
        ax.axvline(pp, color="red", lw=0.8, alpha=0.6)
    ax.set_ylabel("Net polarity")
    ax.set_title(f"Polarity signal  ({len(pol_peaks)} peaks > {pol_pct}th pct)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    speed_full = np.concatenate([speed, [0]])
    ax.plot(x, speed_full * 1000, color="darkgreen", lw=1.2, label="EE speed [mm/frame]")
    for tp in all_trans_tps:
        ax.axvline(tp, color="purple", lw=0.8, alpha=0.5)
    ax.set_ylabel("EE speed [mm/frame]")
    ax.set_title(f"Translation TPs ({len(all_trans_tps)})", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[2]
    ang_full = np.concatenate([np.degrees(ang_speed), [0]])
    ax.plot(x, ang_full, color="sienna", lw=1.2, label="Angular speed [°/frame]")
    for tp in all_rot_tps:
        ax.axvline(tp, color="teal", lw=0.8, alpha=0.5)
    ax.set_ylabel("Angular speed [°/frame]")
    ax.set_title(f"Rotation TPs ({len(all_rot_tps)})", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[3]
    if len(offsets) > 0:
        max_abs = max(abs(offsets.min()), abs(offsets.max()), 1)
        bins = np.arange(-max_abs - 1, max_abs + 2) - 0.5
        ax.hist(offsets, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.plot(offsets, np.zeros_like(offsets) - 0.05, "|", color="steelblue",
                ms=12, markeredgewidth=1.5, clip_on=False)
        ax.axvline(0, color="red", lw=1.5, ls="--", label="exact match")
        ax.axvline(np.median(offsets), color="orange", lw=1.5, ls=":",
                   label=f"median={np.median(offsets):.1f}")
        ax.set_xlabel("Polarity peak − nearest kinematic TP")
        ax.set_ylabel("Count")
        ax.set_title(f"Offset distribution  |  n={len(offsets)}  mean={offsets.mean():.1f}  "
                     f"std={offsets.std():.1f}  median={np.median(offsets):.1f}", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No offsets", transform=ax.transAxes, ha="center", color="gray")

    fig.suptitle(f"Polarity peaks vs kinematic TPs  |  {seq_dir.name}", fontsize=11)
    out = out_dir / "polarity_vs_kinematic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  3. sample
# ══════════════════════════════════════════════════════════════════════

def run_sample(seq_dir: Path, out_dir: Path, idx: Optional[int] = None, split: str = "train") -> None:
    """Show a random input / depth-GT pair from the real dataset."""
    # Import from real_train in the parent directory
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from real_train import RealDataset, DataConfig, D_MAX, ALPHA
    except ImportError:
        print("ERROR: Cannot import real_train. Ensure real_train.py is in the "
              "3d_reconstruction/ directory.")
        return

    def decode_log_depth(d_norm):
        return D_MAX * np.exp(ALPHA * (d_norm - 1.0))

    cfg = DataConfig(seq_len=1, augment=False)
    ds = RealDataset(str(seq_dir), cfg, split=split)
    idx = idx if idx is not None else np.random.randint(len(ds))
    idx = idx % len(ds)
    print(f"Showing sample {idx} / {len(ds) - 1}")

    events, depths, masks = ds[idx]
    voxel = events[0]
    depth_norm = depths[0, 0]
    mask = masks[0, 0]
    depth_m = decode_log_depth(depth_norm)
    event_display = voxel.sum(axis=0)

    frame_idx = int(ds.indices[idx])
    event_frame = None
    ev_h5_path = seq_dir / "hdf5" / "events_cam0.h5"
    if ev_h5_path.exists():
        with h5py.File(ev_h5_path, "r") as f:
            if "events/frames" in f and frame_idx < f["events/frames"].shape[0]:
                event_frame = f["events/frames"][frame_idx]

    n_cols = 4 if event_frame is not None else 3
    fig = plt.figure(figsize=(4 * n_cols + 2, 5))
    fig.suptitle(f"{seq_dir.name} – sample {idx}", fontsize=12)
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.35)
    col = 0

    if event_frame is not None:
        ax = fig.add_subplot(gs[col]); col += 1
        ax.imshow(event_frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title("Event cam frame (HDF5)"); ax.axis("off")

    ax = fig.add_subplot(gs[col]); col += 1
    vmax = np.abs(event_display).max() + 1e-6
    ax.imshow(event_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Events (voxel sum)"); ax.axis("off")

    ax3 = fig.add_subplot(gs[col]); col += 1
    d_masked = np.where(mask > 0, depth_norm, np.nan)
    im3 = ax3.imshow(d_masked, cmap="plasma", vmin=0, vmax=1)
    ax3.set_title("GT depth (log-norm)"); fig.colorbar(im3, ax=ax3, fraction=0.046); ax3.axis("off")

    ax4 = fig.add_subplot(gs[col]); col += 1
    d_m_masked = np.where(mask > 0, depth_m, np.nan)
    im4 = ax4.imshow(d_m_masked, cmap="plasma")
    ax4.set_title("GT depth (metres)"); fig.colorbar(im4, ax=ax4, fraction=0.046, label="m"); ax4.axis("off")

    plt.tight_layout()
    out = out_dir / f"sample_{seq_dir.name}_{idx}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  4. direction_change
# ══════════════════════════════════════════════════════════════════════

def run_direction_change(seq_dir: Path, out_dir: Path, window: int = 20, smooth: int = 3) -> None:
    """Detect polarity reversal and plot relative pose changes."""
    signal = load_polarity_signal(seq_dir)
    n_frames = len(signal)

    s = box_smooth(signal, smooth).astype(np.float32)
    grad = np.gradient(s)
    turning_idx = int(np.argmax(np.abs(grad)))
    smooth_signal = s

    print(f"Frames: {n_frames}, Turning frame: {turning_idx}")

    half = window // 2
    win_start = max(0, turning_idx - half)
    win_end   = min(n_frames, win_start + window)
    win_start = max(0, win_end - window)
    frame_indices = list(range(win_start, win_end))

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]

    T_turning = ee_Ts[turning_idx]
    R_ref = T_turning[:3, :3]
    Z_AXIS = np.array([0.0, 0.0, 1.0])
    offsets_list = [i - turning_idx for i in frame_indices]
    dxs, dys, dzs, d_rots = [], [], [], []
    fxs, fys, fzs = [], [], []

    for idx in frame_indices:
        T_rel = np.linalg.inv(T_turning) @ ee_Ts[idx]
        dt = T_rel[:3, 3]
        R = T_rel[:3, :3]
        cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_a)))
        dxs.append(dt[0] * 1000); dys.append(dt[1] * 1000); dzs.append(dt[2] * 1000)
        d_rots.append(angle_deg)
        R_i = ee_Ts[idx, :3, :3]
        R_rel2 = R_ref.T @ R_i
        facing = R_rel2 @ Z_AXIS
        fxs.append(facing[0]); fys.append(facing[1]); fzs.append(facing[2])

    offsets_arr = np.array(offsets_list)
    dxs, dys, dzs, d_rots = map(np.array, [dxs, dys, dzs, d_rots])
    fxs, fys, fzs = map(np.array, [fxs, fys, fzs])

    fig = plt.figure(figsize=(max(10, len(frame_indices) * 0.7), 12))
    gs_fig = fig.add_gridspec(4, 1, hspace=0.50)
    x_all = np.arange(n_frames)

    ax1 = fig.add_subplot(gs_fig[0])
    ax1.plot(x_all, signal, color="steelblue", lw=0.8, alpha=0.5, label="raw")
    ax1.plot(x_all, smooth_signal, color="steelblue", lw=1.5, label=f"smoothed (k={smooth})")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.axvspan(win_start, win_end - 1, alpha=0.12, color="orange", label="window")
    ax1.axvline(turning_idx, color="red", lw=1.5, ls="--", label=f"turning (#{turning_idx})")
    ax1.set_xlabel("Frame index"); ax1.set_ylabel("Net polarity")
    ax1.set_title("Event polarity signal", fontsize=10); ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs_fig[1])
    ax2.plot(offsets_arr, dxs, "r.-", lw=1.4, ms=5, label="dx")
    ax2.plot(offsets_arr, dys, "g.-", lw=1.4, ms=5, label="dy")
    ax2.plot(offsets_arr, dzs, "b.-", lw=1.4, ms=5, label="dz")
    ax2.axhline(0, color="gray", lw=0.8, ls="--"); ax2.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6)
    ax2.set_xlabel("Frame offset"); ax2.set_ylabel("Δ translation [mm]")
    ax2.set_title("Relative translation from turning frame", fontsize=10); ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs_fig[2])
    ax3.plot(offsets_arr, fxs, "r.-", lw=1.4, ms=5, label="facing x")
    ax3.plot(offsets_arr, fys, "g.-", lw=1.4, ms=5, label="facing y")
    ax3.plot(offsets_arr, fzs, "b.-", lw=1.4, ms=5, label="facing z")
    ax3.axhline(0, color="gray", lw=0.8, ls="--"); ax3.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6)
    ax3.set_ylim(-1.05, 1.05); ax3.set_xlabel("Frame offset")
    ax3.set_ylabel("Facing direction"); ax3.set_title("Relative facing (+Z of EE)", fontsize=10)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs_fig[3])
    ax4.plot(offsets_arr, d_rots, "k.-", lw=1.4, ms=5)
    ax4.axvline(0, color="red", lw=1.5, ls="--", alpha=0.6)
    ax4.set_xlabel("Frame offset"); ax4.set_ylabel("Δ rotation [°]")
    ax4.set_title("Geodesic rotation distance from turning frame", fontsize=10)
    ax4.set_ylim(bottom=0)

    fig.suptitle(f"Direction change  |  {seq_dir.name}  |  turning frame: {turning_idx}", fontsize=10)
    out = out_dir / "direction_change.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  5. least_active
# ══════════════════════════════════════════════════════════════════════

def run_least_active(seq_dir: Path, out_dir: Path, window: int = 40, smooth: int = 3) -> None:
    """Find the quietest event frame and show surrounding context."""
    activity, polarity = load_activity_and_polarity(seq_dir)
    n = len(activity)

    with h5py.File(seq_dir / "hdf5" / "poses.h5", "r") as f:
        ee_Ts = f["ee_T"][:]
    positions = ee_Ts[:, :3, 3]
    rotations = ee_Ts[:, :3, :3]

    trans_speed = np.zeros(n, dtype=np.float32)
    trans_speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 1000
    ang_speed_deg = np.degrees(rotation_angle_per_frame(rotations))

    quiet_idx = int(np.argmin(box_smooth(activity, smooth)))
    print(f"Quietest frame: {quiet_idx}  (activity={activity[quiet_idx]:.5f})")

    half = window // 2
    win_start = max(0, quiet_idx - half)
    win_end   = min(n, win_start + window)
    win_start = max(0, win_end - window)
    win_x = np.arange(win_start, win_end)
    off_x = win_x - quiet_idx

    act_s   = box_smooth(activity[win_start:win_end], smooth)
    pol_s   = box_smooth(polarity[win_start:win_end], smooth)
    trans_s = box_smooth(trans_speed[win_start:win_end], smooth)
    ang_s   = box_smooth(ang_speed_deg[win_start:win_end], smooth)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.45)

    ax = axes[0]
    ax.plot(np.arange(n), activity, color="steelblue", lw=0.8, alpha=0.5, label="raw")
    ax.plot(np.arange(n), box_smooth(activity, smooth), color="steelblue", lw=1.5, label=f"smoothed (k={smooth})")
    ax.axvspan(win_start, win_end - 1, alpha=0.12, color="orange", label="window")
    ax.axvline(quiet_idx, color="red", lw=1.5, ls="--", label=f"quietest #{quiet_idx}")
    ax.set_xlabel("Frame index"); ax.set_ylabel("Event activity"); ax.set_title("Full recording", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(off_x, act_s, color="steelblue", lw=1.5, label="event activity")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7)
    ax.set_ylabel("Event activity", color="steelblue")
    ax_r = ax.twinx()
    ax_r.plot(off_x, pol_s, color="darkorange", lw=1.5, ls="--", label="net polarity")
    ax_r.axhline(0, color="gray", lw=0.6, ls=":"); ax_r.set_ylabel("Net polarity", color="darkorange")
    ax.set_xlabel("Frame offset"); ax.set_title("Activity & polarity around quiet point", fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    ax = axes[2]
    ax.plot(off_x, trans_s, color="forestgreen", lw=1.5, label="translation [mm/frame]")
    ax.axvline(0, color="red", lw=1.5, ls="--", alpha=0.7)
    ax.set_ylabel("Translation speed", color="forestgreen")
    ax_r2 = ax.twinx()
    ax_r2.plot(off_x, ang_s, color="tomato", lw=1.5, ls="--", label="angular [°/frame]")
    ax_r2.set_ylabel("Angular speed", color="tomato")
    ax.set_xlabel("Frame offset"); ax.set_title("Speeds around quiet point", fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    fig.suptitle(f"Least-active frame  |  {seq_dir.name}  |  #{quiet_idx}", fontsize=10)
    out = out_dir / "least_active.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  6. poses
# ══════════════════════════════════════════════════════════════════════

def run_poses(seq_dir: Path, out_dir: Path, calib_dir: str = "camera_data",
              n_samples: int = 6, seed: Optional[int] = None) -> None:
    """Visualize camera poses + corresponding RGB images."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.gridspec import GridSpec

    calib_dir_p = Path(calib_dir)
    AXIS_LEN = 0.03; ARROW_LEN = 0.06

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
#  7. raw_poses
# ══════════════════════════════════════════════════════════════════════

def run_raw_poses(seq_dir: Path, out_dir: Path) -> None:
    """Plot raw pose data: position, orientation, joints, rotation speed, gripper, rate."""
    raw_path = seq_dir / "hdf5" / "raw_poses.h5"
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw_poses.h5 in {seq_dir / 'hdf5'}")

    with h5py.File(raw_path, "r") as f:
        t_ns      = f["t_ns"][:]
        ee_T      = f["ee_T"][:]
        joint_pos = f["joint_positions"][:]
        gripper_q = f["gripper_q"][:]

    P = len(t_ns)
    t_sec = (t_ns - t_ns[0]) / 1e9
    pos = ee_T[:, :3, 3]
    R = ee_T[:, :3, :3]
    yaw, pitch, roll = rotation_matrix_to_euler_zyx(R)
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    dt = np.diff(t_sec); dt[dt == 0] = 1e-6
    pose_rate = 1.0 / dt
    rot_vel = np.zeros(P, dtype=np.float64)
    rot_vel[1:] = np.diff(z_angle) / dt
    rot_vel[np.abs(rot_vel) > 20.0] = 0.0

    print(f"Raw poses: {P}, Duration: {t_sec[-1]:.2f} s, Median rate: {np.median(pose_rate):.1f} Hz")

    fig = plt.figure(figsize=(14, 19))
    gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0.45)

    ax = fig.add_subplot(gs[0])
    ax.plot(t_sec, pos[:, 0], lw=1.0, label="X"); ax.plot(t_sec, pos[:, 1], lw=1.0, label="Y")
    ax.plot(t_sec, pos[:, 2], lw=1.0, label="Z")
    ax.set_ylabel("Position (m)"); ax.set_title("EE Position", fontsize=11)
    ax.legend(fontsize=8); ax.set_xlabel("Time (s)"); ax.grid(True, lw=0.4, alpha=0.5)

    ax = fig.add_subplot(gs[1])
    ax.plot(t_sec, yaw, lw=1.0, label="Yaw"); ax.plot(t_sec, pitch, lw=1.0, label="Pitch")
    ax.plot(t_sec, roll, lw=1.0, label="Roll")
    ax.set_ylabel("Angle (°)"); ax.set_title("EE Orientation — ZYX Euler", fontsize=11)
    ax.legend(fontsize=8); ax.set_xlabel("Time (s)"); ax.grid(True, lw=0.4, alpha=0.5)

    ax = fig.add_subplot(gs[2])
    for j in range(joint_pos.shape[1]):
        ax.plot(t_sec, np.rad2deg(joint_pos[:, j]), lw=0.8, label=f"J{j+1}")
    ax.set_ylabel("Angle (°)"); ax.set_title("Joint Positions", fontsize=11)
    ax.legend(fontsize=7, ncol=4); ax.set_xlabel("Time (s)"); ax.grid(True, lw=0.4, alpha=0.5)

    ax = fig.add_subplot(gs[3])
    ax.plot(t_sec, np.rad2deg(rot_vel), lw=0.8, color="#e08020", label="Z-rot speed")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("Speed (°/s)"); ax.set_title("EE Rotation Speed (Z-axis)", fontsize=11)
    ax.set_xlabel("Time (s)"); ax.grid(True, lw=0.4, alpha=0.5)

    ax = fig.add_subplot(gs[4])
    ax.plot(t_sec, gripper_q, lw=1.0, color="#555555")
    ax.set_ylabel("Gripper opening"); ax.set_title("Gripper", fontsize=11)
    ax.set_xlabel("Time (s)"); ax.grid(True, lw=0.4, alpha=0.5)

    ax = fig.add_subplot(gs[5])
    ax.plot(t_sec[1:], pose_rate, lw=0.7, color="#888888")
    ax.axhline(np.median(pose_rate), color="red", lw=1.0, ls="--",
               label=f"median = {np.median(pose_rate):.1f} Hz")
    ax.set_ylabel("Rate (Hz)"); ax.set_title("Pose publish rate", fontsize=11)
    ax.legend(fontsize=8); ax.set_xlabel("Time (s)"); ax.grid(True, lw=0.4, alpha=0.5)

    fig.suptitle(f"Raw Poses — {seq_dir.name}  ({P} poses, {t_sec[-1]:.1f} s)",
                 fontsize=13, fontweight="bold", y=0.98)
    out = out_dir / "raw_poses.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  8. turning_point
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
#  9. voxels
# ══════════════════════════════════════════════════════════════════════

def run_voxels(seq_dir: Path, out_dir: Path, indices: Optional[list] = None,
               n_random: int = 3, cam: int = 0) -> None:
    """Visualize precomputed voxel grids + event / RGB frames."""
    voxels_dir   = seq_dir / "events" / f"voxels_cam{cam}"
    events_h5    = seq_dir / "hdf5" / f"events_cam{cam}.h5"
    realsense_h5 = seq_dir / "hdf5" / "realsense.h5"

    if not voxels_dir.exists():
        sys.exit(f"Voxels directory not found: {voxels_dir}")

    voxel_files = sorted(voxels_dir.glob("voxel_*.npy"))
    n_frames = len(voxel_files)
    if n_frames == 0:
        sys.exit(f"No voxel files in {voxels_dir}")

    if indices is not None:
        idx_list = [i % n_frames for i in indices]
    else:
        rng = np.random.default_rng()
        idx_list = sorted(rng.choice(n_frames, size=min(n_random, n_frames), replace=False).tolist())

    sample = np.load(voxel_files[0])
    num_bins = sample.shape[0]
    n_rows = len(idx_list)
    n_cols = 1 + num_bins + 1 + 1
    col_w = 2.2

    fig = plt.figure(figsize=(col_w * n_cols + 0.5, col_w * n_rows + 1.0))
    fig.suptitle(f"{seq_dir.name} – cam{cam} – indices {idx_list}", fontsize=11)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.08)

    for row, idx in enumerate(idx_list):
        voxel = np.load(voxels_dir / f"voxel_{idx:06d}.npy")
        voxel_sum = voxel.sum(axis=0)

        # Event frame
        ev_frame = None
        if events_h5.exists():
            with h5py.File(events_h5, "r") as f:
                if "events/frames" in f and idx < f["events/frames"].shape[0]:
                    ev_frame = f["events/frames"][idx]

        # RGB frame
        rgb_frame = None
        if realsense_h5.exists():
            with h5py.File(realsense_h5, "r") as f:
                if "rgb" in f and idx < f["rgb"].shape[0]:
                    rgb_frame = f["rgb"][idx]

        col = 0
        ax = fig.add_subplot(gs[row, col]); col += 1
        if ev_frame is not None:
            ax.imshow(ev_frame, cmap="gray", vmin=0, vmax=255)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        if row == 0: ax.set_title("event\nframe", fontsize=8)
        ax.set_ylabel(f"idx {idx}", fontsize=8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        nonzero = voxel[voxel != 0]
        scale = (3.0 * nonzero.std()) if len(nonzero) > 0 and nonzero.std() > 0 else 1.0
        for b in range(num_bins):
            ax = fig.add_subplot(gs[row, col]); col += 1
            gray = np.clip(voxel[b] / scale * 127 + 128, 0, 255).astype(np.uint8)
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            if row == 0: ax.set_title(f"bin {b}", fontsize=8)
            ax.set_yticks([]); ax.set_xticks([])

        ax = fig.add_subplot(gs[row, col]); col += 1
        nz_sum = voxel_sum[voxel_sum != 0]
        sc_sum = (3.0 * nz_sum.std()) if len(nz_sum) > 0 and nz_sum.std() > 0 else 1.0
        gray_sum = np.clip(voxel_sum / sc_sum * 127 + 128, 0, 255).astype(np.uint8)
        ax.imshow(gray_sum, cmap="gray", vmin=0, vmax=255)
        if row == 0: ax.set_title("voxel\nsum", fontsize=8)
        ax.set_yticks([]); ax.set_xticks([])

        ax = fig.add_subplot(gs[row, col]); col += 1
        if rgb_frame is not None:
            ax.imshow(rgb_frame)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        if row == 0: ax.set_title("RGB", fontsize=8)
        ax.set_yticks([]); ax.set_xticks([])

    plt.tight_layout()
    out = out_dir / f"voxels_{seq_dir.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  10. white_mask
# ══════════════════════════════════════════════════════════════════════

def run_white_mask(seq_dir: Path, out_dir: Path, indices: Optional[list] = None,
                   n_random: int = 5, cam: int = 0, white_thresh: int = 200) -> None:
    """Visualize white-pixel mask and its projection onto the event plane."""
    realsense_h5 = seq_dir / "hdf5" / "realsense.h5"
    proj_rgb_h5  = seq_dir / "hdf5" / "rgb_in_event_frame.h5"
    voxels_dir   = seq_dir / "events" / f"voxels_cam{cam}"
    if not voxels_dir.exists():
        voxels_dir = seq_dir / "events" / "voxels"

    if not realsense_h5.exists():
        sys.exit(f"realsense.h5 not found: {realsense_h5}")
    if not proj_rgb_h5.exists():
        sys.exit(f"rgb_in_event_frame.h5 not found: {proj_rgb_h5}")
    if not voxels_dir.exists():
        sys.exit(f"No voxels directory: {voxels_dir}")

    n_frames = len(list(voxels_dir.glob("voxel_*.npy")))
    if n_frames == 0:
        sys.exit(f"No voxel files in {voxels_dir}")

    if indices is not None:
        idx_list = [i % n_frames for i in indices]
    else:
        rng = np.random.default_rng()
        idx_list = sorted(rng.choice(n_frames, size=min(n_random, n_frames), replace=False).tolist())

    def _white_mask(rgb, thresh):
        return np.all(rgb > thresh, axis=-1)

    def _render_overlay(bg, mask):
        out = bg.copy().astype(np.float32) / 255.0
        out[~mask] *= 0.5
        out[mask] = [1.0, 0.0, 0.0]
        return (out * 255).astype(np.uint8)

    def _event_to_rgb(vsum):
        v = vsum.astype(np.float32)
        vmax = np.abs(v).max()
        if vmax > 0: v = v / vmax
        img = ((v + 1.0) * 0.5 * 255).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)

    n_rows = len(idx_list)
    fig = plt.figure(figsize=(14, 3.0 * n_rows + 0.6))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.05, wspace=0.05,
                           left=0.02, right=0.98, top=0.94, bottom=0.02)

    col_titles = ["Original RGB", f"White mask on RGB (>{white_thresh})",
                  "Event (voxel sum)", f"White mask on event (>{white_thresh})"]
    for ci, t in enumerate(col_titles):
        fig.add_subplot(gs[0, ci]).set_title(t, fontsize=8, pad=3)

    for row, idx in enumerate(idx_list):
        with h5py.File(realsense_h5, "r") as f:
            rgb = f["rgb"][idx] if idx < f["rgb"].shape[0] else None
        with h5py.File(proj_rgb_h5, "r") as f:
            proj_rgb = f["rgb"][idx] if "rgb" in f and idx < f["rgb"].shape[0] else None
        vox_path = voxels_dir / f"voxel_{idx:06d}.npy"
        vox_sum = np.load(vox_path).sum(axis=0) if vox_path.exists() else None

        if rgb is None or proj_rgb is None or vox_sum is None:
            continue

        white_rgb   = _white_mask(rgb, white_thresh)
        white_event = _white_mask(proj_rgb, white_thresh)
        overlay_rgb = _render_overlay(rgb, white_rgb)
        event_img   = _event_to_rgb(vox_sum)
        overlay_ev  = _render_overlay(event_img, white_event)

        for ci, panel in enumerate([rgb, overlay_rgb, event_img, overlay_ev]):
            ax = fig.add_subplot(gs[row, ci])
            ax.imshow(panel); ax.set_xticks([]); ax.set_yticks([])
            if ci == 0: ax.set_ylabel(f"idx={idx}", fontsize=7)
            if ci == 1: ax.set_xlabel(f"masked: {100*white_rgb.mean():.1f}%", fontsize=6)
            if ci == 3: ax.set_xlabel(f"masked: {100*white_event.mean():.1f}%", fontsize=6)

    fig.suptitle(f"White-pixel mask projection  |  {seq_dir.name}", fontsize=10, y=0.98)
    out = out_dir / f"white_mask_{seq_dir.name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified data analysis — generate image plots from recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Analysis to run")

    # -- polarity_speed --
    p = sub.add_parser("polarity_speed", help="Overlay polarity, translation, angular speed")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--smooth", type=int, default=5)

    # -- polarity_kinematic --
    p = sub.add_parser("polarity_kinematic", help="Polarity peaks vs kinematic TPs")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--pol_pct", type=float, default=85)
    p.add_argument("--kin_pct", type=float, default=70)
    p.add_argument("--min_dist", type=int, default=5)

    # -- sample --
    p = sub.add_parser("sample", help="Show random dataset sample")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--idx", type=int, default=None)
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])

    # -- direction_change --
    p = sub.add_parser("direction_change", help="Detect polarity reversal & plot poses")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--smooth", type=int, default=3)

    # -- least_active --
    p = sub.add_parser("least_active", help="Find quietest event frame")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=40)
    p.add_argument("--smooth", type=int, default=3)

    # -- poses --
    p = sub.add_parser("poses", help="Camera poses + RGB images")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--calib_dir", type=str, default="camera_data")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=None)

    # -- raw_poses --
    p = sub.add_parser("raw_poses", help="Plot raw pose data")
    p.add_argument("--data_dir", type=str, required=True)

    # -- turning_point --
    p = sub.add_parser("turning_point", help="Frames around sharpest EE direction reversal")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--smooth", type=int, default=1)

    # -- voxels --
    p = sub.add_parser("voxels", help="Voxel grids + event/RGB frames")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--indices", type=int, nargs="+", default=None)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--cam", type=int, default=0)

    # -- white_mask --
    p = sub.add_parser("white_mask", help="White-pixel mask projection")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--indices", type=int, nargs="+", default=None)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--white_thresh", type=int, default=200)

    # -- all --
    p = sub.add_parser("all", help="Run every analysis")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--smooth", type=int, default=5)
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

    if cmd == "polarity_speed":
        run_polarity_speed(seq_dir, out_dir, smooth=args.smooth)
    elif cmd == "polarity_kinematic":
        run_polarity_kinematic(seq_dir, out_dir, smooth=args.smooth,
                               pol_pct=args.pol_pct, kin_pct=args.kin_pct, min_dist=args.min_dist)
    elif cmd == "sample":
        run_sample(seq_dir, out_dir, idx=args.idx, split=args.split)
    elif cmd == "direction_change":
        run_direction_change(seq_dir, out_dir, window=args.window, smooth=args.smooth)
    elif cmd == "least_active":
        run_least_active(seq_dir, out_dir, window=args.window, smooth=args.smooth)
    elif cmd == "poses":
        run_poses(seq_dir, out_dir, calib_dir=args.calib_dir, n_samples=args.n, seed=args.seed)
    elif cmd == "raw_poses":
        run_raw_poses(seq_dir, out_dir)
    elif cmd == "turning_point":
        run_turning_point(seq_dir, out_dir, window=args.window, smooth=args.smooth)
    elif cmd == "voxels":
        run_voxels(seq_dir, out_dir, indices=args.indices, n_random=args.n, cam=args.cam)
    elif cmd == "white_mask":
        run_white_mask(seq_dir, out_dir, indices=args.indices, n_random=args.n,
                       cam=args.cam, white_thresh=args.white_thresh)
    elif cmd == "all":
        print(f"Running all analyses for {seq_dir.name} → {out_dir}\n")
        errors = []
        for name, fn in [
            ("polarity_speed",     lambda: run_polarity_speed(seq_dir, out_dir, smooth=args.smooth)),
            ("polarity_kinematic", lambda: run_polarity_kinematic(seq_dir, out_dir, smooth=args.smooth)),
            ("direction_change",   lambda: run_direction_change(seq_dir, out_dir)),
            ("least_active",       lambda: run_least_active(seq_dir, out_dir)),
            ("poses",              lambda: run_poses(seq_dir, out_dir, calib_dir=args.calib_dir)),
            ("raw_poses",          lambda: run_raw_poses(seq_dir, out_dir)),
            ("turning_point",      lambda: run_turning_point(seq_dir, out_dir)),
            ("voxels",             lambda: run_voxels(seq_dir, out_dir)),
            ("white_mask",         lambda: run_white_mask(seq_dir, out_dir)),
            ("sample",             lambda: run_sample(seq_dir, out_dir)),
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
