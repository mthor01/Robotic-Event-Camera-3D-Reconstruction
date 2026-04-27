#!/usr/bin/env python3
"""
Unified pose temporal alignment analysis script.

Merges:
  frame_level  – Frame-level temporal alignment (rotation speed vs event activity)
  voxel_level  – Voxel-bin level temporal alignment (5× higher resolution)
  verify       – Verify interpolated poses vs camera frame timestamps
  all          – Run every analysis above

Output images are saved to  <data_dir>/pose_plots/  by default.

NOTE: The TemporalAlignmentAgent rotates the EE in place about the Z axis
(±45° oscillation at a fixed position).  All analyses in this script
extract Z-rotation via atan2(R[1,0], R[0,0]) and compare rotation speed
against event activity — they are fully adapted for the rotate-in-place
motion pattern and do NOT depend on any translational movement.

Usage examples:
    python pose_time_analysis.py frame_level  --data_dir data/real/temporal_check
    python pose_time_analysis.py voxel_level  --data_dir data/real/temporal_check
    python pose_time_analysis.py verify       --data_dir data/real/1
    python pose_time_analysis.py all          --data_dir data/real/temporal_check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction_config import FPS as DEFAULT_FPS


# ══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════

def ensure_out_dir(data_dir: Path) -> Path:
    """Return <data_dir>/pose_plots, creating it if necessary."""
    d = data_dir / "pose_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.astype(np.float64).copy()
    kernel = np.ones(k) / k
    return np.convolve(arr.astype(np.float64), kernel, mode="same")


def find_troughs(sig: np.ndarray, min_prominence_frac: float = 0.15) -> np.ndarray:
    """Find troughs using scipy find_peaks on inverted signal."""
    if len(sig) < 3:
        return np.array([], dtype=np.float64)
    sig_range = float(sig.max() - sig.min())
    if sig_range == 0:
        return np.array([], dtype=np.float64)
    peaks, _ = find_peaks(-sig, prominence=min_prominence_frac * sig_range)
    return peaks.astype(np.float64)


def unit_norm(a: np.ndarray) -> np.ndarray:
    mx = np.abs(a).max()
    return a / mx if mx > 0 else a


# ══════════════════════════════════════════════════════════════════════
#  1. frame_level  (from analyze_temporal_alignment.py)
# ══════════════════════════════════════════════════════════════════════

def run_frame_level(seq_dir: Path, out_dir: Path, smooth_k: int = 3) -> None:
    """Frame-level temporal alignment: rotation speed vs event activity.

    The robot holds a fixed position and oscillates its Z-rotation.
    Every direction reversal produces a dip in event activity (the arm
    is momentarily still).  This analysis detects those dips in both
    the rotation speed and event activity signals, matches them, and
    computes per-pair temporal offsets.
    """
    hdf5_dir = seq_dir / "hdf5"

    # Load poses
    poses_path = hdf5_dir / "poses.h5"
    if not poses_path.exists():
        raise FileNotFoundError(f"No poses.h5 in {hdf5_dir}")
    with h5py.File(poses_path, "r") as f:
        ee_T = f["ee_T"][:]

    # Load frame timestamps from realsense.h5 (poses.h5 does not store t_ns)
    rs_path = hdf5_dir / "realsense.h5"
    if rs_path.exists():
        with h5py.File(rs_path, "r") as f:
            # Prefer hardware-anchored timestamps (cleaner inter-frame spacing)
            if "t_hw_as_sys_ns" in f:
                t_ns = f["t_hw_as_sys_ns"][:]
            else:
                t_ns = f["t_sys_ns"][:]
    else:
        t_ns = np.arange(len(ee_T), dtype=np.float64) * (1e9 / DEFAULT_FPS)

    N = len(ee_T)
    R = ee_T[:, :3, :3]
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    t_sec = (t_ns - t_ns[0]) / 1e9 if t_ns.dtype != np.float64 or t_ns.max() > 1e6 else t_ns.copy()
    dt = np.diff(t_sec); dt[dt == 0] = 1.0
    rot_vel = np.zeros(N, dtype=np.float64)
    rot_vel[1:] = np.diff(z_angle) / dt
    rot_vel[np.abs(rot_vel) > 10.0] = 0.0

    # Load event activity
    ev_h5 = hdf5_dir / "events_cam0.h5"
    if not ev_h5.exists():
        raise FileNotFoundError(f"No events_cam0.h5 in {hdf5_dir}")
    with h5py.File(ev_h5, "r") as f:
        ev_frames = f["events/frames"][:]
    event_activity = ev_frames.astype(np.float64).std(axis=(1, 2))
    M = len(event_activity)

    L = min(N, M)
    START = min(100, L)
    END = max(START, L - 100)
    rot_speed = np.abs(rot_vel[START:END])
    event_activity = event_activity[START:END]
    t_sec_trim = t_sec[START:END]
    z_angle_plot = z_angle[START:END]
    L = len(rot_speed)

    speed_centered = rot_speed - rot_speed.mean()
    evcount_centered = event_activity - event_activity.mean()

    vel_smooth = box_smooth(speed_centered, smooth_k)
    ev_smooth  = box_smooth(evcount_centered, smooth_k)

    spikes_vel = find_troughs(vel_smooth)
    spikes_ev  = find_troughs(ev_smooth)

    # Estimate expected inter-trough spacing to cap match distance
    n_troughs = max(len(spikes_vel), len(spikes_ev), 1)
    max_match_dist = max(3.0, L / n_troughs * 0.4)  # 40% of avg spacing, min 3 frames

    # Match spikes — symmetric greedy (sort all pairs by distance, match closest first)
    offsets_frames = []
    matched_vel, matched_ev = [], []
    if len(spikes_vel) > 0 and len(spikes_ev) > 0:
        candidates = sorted(
            (abs(float(a) - float(b)), ia, ib)
            for ia, a in enumerate(spikes_vel)
            for ib, b in enumerate(spikes_ev)
        )
        used_vel, used_ev = set(), set()
        for dist, ia, ib in candidates:
            if dist > max_match_dist:
                break
            if ia not in used_vel and ib not in used_ev:
                used_vel.add(ia); used_ev.add(ib)
                offsets_frames.append(float(spikes_ev[ib] - spikes_vel[ia]))
                matched_vel.append(float(spikes_vel[ia]))
                matched_ev.append(float(spikes_ev[ib]))

    offsets_frames = np.array(offsets_frames) if offsets_frames else np.array([])
    int_offsets = np.round(offsets_frames).astype(int) if len(offsets_frames) > 0 else np.array([], dtype=int)
    frame_dt = np.median(np.diff(t_sec_trim)) if len(t_sec_trim) > 1 else 1.0 / 30.0
    offsets_ms = offsets_frames * frame_dt * 1000.0

    # Keep only pairs with |offset| < 100 ms
    if len(offsets_ms) > 0:
        mask = np.abs(offsets_ms) < 100.0
        offsets_frames = offsets_frames[mask]
        int_offsets = int_offsets[mask]
        offsets_ms = offsets_ms[mask]
        matched_vel = [v for v, m in zip(matched_vel, mask) if m]
        matched_ev  = [e for e, m in zip(matched_ev,  mask) if m]

    # Print summary
    print("\n" + "=" * 60)
    print("  FRAME-LEVEL TEMPORAL ALIGNMENT")
    print("=" * 60)
    print(f"  Recording:       {seq_dir}")
    print(f"  Pose frames: {N}    Event frames: {M}    Common: {L}")
    print(f"  Speed spikes: {len(spikes_vel)}    Event spikes: {len(spikes_ev)}")
    if len(int_offsets) > 0:
        for i, (ov, oe, off_f, off_m) in enumerate(zip(matched_vel, matched_ev, int_offsets, offsets_ms)):
            print(f"    pair {i}: speed@{ov:.0f} event@{oe:.0f} offset={off_f:+d}fr ({off_m:+.1f}ms)")
        print(f"  Mean: {int_offsets.mean():+.1f}fr  Median: {np.median(int_offsets):+.1f}fr  Std: {int_offsets.std():.1f}fr")
    print("=" * 60)

    # Figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1.2], hspace=0.35, wspace=0.30)
    frames = np.arange(L)
    color_pos, color_ev = "#2066a8", "#d6604d"

    # Panel 1: Z-rotation + event activity
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(frames, np.rad2deg(z_angle_plot), color=color_pos, lw=1.4, label="EE Z-rotation")
    ax1.set_ylabel("Z-rotation (°)", color=color_pos)
    ax1r = ax1.twinx()
    ax1r.plot(frames, box_smooth(event_activity, smooth_k), color=color_ev, lw=1.0, alpha=0.8, label="event activity")
    ax1r.set_ylabel("Event activity (pixel std)", color=color_ev)
    ax1.set_title("EE Z-rotation & event activity", fontsize=11)
    ax1.set_xlabel("Frame index")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, fontsize=8)

    # Panel 2: normalised overlay
    ax2 = fig.add_subplot(gs[1, :])
    vn = unit_norm(vel_smooth); en = unit_norm(ev_smooth)
    ax2.plot(frames, vn, color=color_pos, lw=1.3, label="Rotation speed (norm)")
    ax2.plot(frames, en, color=color_ev, lw=1.3, alpha=0.8, label="Event activity (norm)")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    for sv in spikes_vel:
        i = int(min(round(sv), L - 1)); ax2.plot(sv, vn[i], "v", color=color_pos, ms=7, zorder=5)
    for se in spikes_ev:
        i = int(min(round(se), L - 1)); ax2.plot(se, en[i], "v", color=color_ev, ms=7, zorder=5)
    ax2.set_ylabel("Normalised"); ax2.set_xlabel("Frame index")
    ax2.set_title("Rotation speed vs event activity (▼ = troughs)", fontsize=11)
    ax2.legend(fontsize=8)

    # Panel 3: histogram
    ax3 = fig.add_subplot(gs[2, 0])  # noqa
    if len(int_offsets) > 0:
        lo, hi = int_offsets.min(), int_offsets.max()
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        ax3.hist(int_offsets, bins=bins, color="#8da0cb", edgecolor="k", linewidth=0.8, rwidth=0.85)
        ax3.axvline(0, color="red", lw=1.2, ls="--", label="zero")
        ax3.axvline(float(np.median(int_offsets)), color="orange", lw=1.2, ls="-",
                    label=f"median={np.median(int_offsets):+.0f}fr")
        ax3.set_xlabel("Frame offset"); ax3.set_ylabel("Count")
        ax3.set_title("Offset distribution", fontsize=10); ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No reversals", ha="center", va="center", transform=ax3.transAxes, color="red")

    # Panel 4: offset over time
    ax4 = fig.add_subplot(gs[2, 1])
    if len(int_offsets) > 0:
        dot_colors = ["#4daf4a" if o == 0 else ("#2066a8" if abs(o) <= 1 else "#ff7f00") for o in int_offsets]
        ax4.scatter(matched_vel, int_offsets, color=dot_colors, edgecolors="k", linewidths=0.5, s=40, zorder=3)
        ax4.axhline(0, color="gray", lw=0.5, ls=":")
        ax4.axhline(float(np.median(int_offsets)), color="red", lw=1.0, ls="--")
        ax4.set_xlabel("Frame of speed spike"); ax4.set_ylabel("Offset (frames)")
        ax4.set_title("Offset over time", fontsize=10)
        med = int(np.median(int_offsets))
        label = "✓ GOOD" if abs(med) == 0 else ("~ OK (±1)" if abs(med) == 1 else f"✗ OFF ({med:+d}fr)")
        ax4.set_title(f"Offset over time  |  n={len(int_offsets)}  med={med:+d}fr  {label}", fontsize=9)

    fig.suptitle("Frame-Level Temporal Alignment — Rotation Speed vs Event Activity",
                 fontsize=13, fontweight="bold", y=0.98)
    out = out_dir / "temporal_alignment_frame.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  2. voxel_level  (from analyze_temporal_alignment_voxel.py)
# ══════════════════════════════════════════════════════════════════════

def run_voxel_level(seq_dir: Path, out_dir: Path, smooth_k: int = 5) -> None:
    """Voxel-bin level temporal alignment at ~5× higher resolution."""
    hdf5_dir = seq_dir / "hdf5"

    # Raw poses
    raw_path = hdf5_dir / "raw_poses.h5"
    if not raw_path.exists():
        raise FileNotFoundError(f"No raw_poses.h5 in {hdf5_dir}")
    with h5py.File(raw_path, "r") as f:
        pose_t_ns = f["t_ns"][:]
        pose_ee_T = f["ee_T"][:]
    P = len(pose_t_ns)

    # Metadata
    meta_path = hdf5_dir / "metadata.h5"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.h5 in {hdf5_dir}")
    with h5py.File(meta_path, "r") as f:
        event_logging_start_ns = int(f.attrs["event_logging_start_ns"])

    # Event frame timestamps
    ev_h5 = hdf5_dir / "events_cam0.h5"
    if not ev_h5.exists():
        raise FileNotFoundError(f"No events_cam0.h5 in {hdf5_dir}")
    with h5py.File(ev_h5, "r") as f:
        t_start_us = f["events/t_ev_start_us"][:]
        t_end_us   = f["events/t_ev_end_us"][:]
    N_frames = len(t_start_us)

    # Voxels — stored as a single HDF5 file (shape: N, bins, H, W)
    voxels_h5 = seq_dir / "events" / "voxels_cam0.h5"
    if not voxels_h5.exists():
        raise FileNotFoundError(f"No voxels file: {voxels_h5}")

    with h5py.File(voxels_h5, "r") as vf:
        voxels_ds = vf["voxels"]
        n_voxel_frames = min(voxels_ds.shape[0], N_frames)
        num_bins = voxels_ds.shape[1]

        bin_t_us, bin_activity = [], []
        for fi in range(n_voxel_frames):
            voxel = voxels_ds[fi]  # (bins, H, W)
            t0, t1 = float(t_start_us[fi]), float(t_end_us[fi])
            for b in range(num_bins):
                frac = (b + 0.5) / num_bins
                bin_t_us.append(t0 + frac * (t1 - t0))
                bin_activity.append(float(np.std(voxel[b])))

    bin_t_us = np.array(bin_t_us)
    bin_activity = np.array(bin_activity)
    N_bins = len(bin_t_us)

    ev_elapsed_ns = (bin_t_us - t_start_us[0]).astype(np.float64) * 1000.0
    pose_elapsed_ns = (pose_t_ns - event_logging_start_ns).astype(np.float64)

    # Nearest pose matching
    indices = np.searchsorted(pose_elapsed_ns, ev_elapsed_ns, side="left")
    indices = np.clip(indices, 0, P - 1)
    left = np.clip(indices - 1, 0, P - 1)
    use_left = np.abs(pose_elapsed_ns[left] - ev_elapsed_ns) < np.abs(pose_elapsed_ns[indices] - ev_elapsed_ns)
    indices[use_left] = left[use_left]

    matched_ee_T = pose_ee_T[indices]
    R = matched_ee_T[:, :3, :3]
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    bin_t_sec = (ev_elapsed_ns - ev_elapsed_ns[0]) / 1e9
    dt = np.diff(bin_t_sec); dt[dt == 0] = 1e-6
    rot_vel = np.zeros(N_bins, dtype=np.float64)
    rot_vel[1:] = np.diff(z_angle) / dt
    rot_vel[np.abs(rot_vel) > 10.0] = 0.0

    TRIM = 100 * num_bins
    START = min(TRIM, N_bins)
    END = max(START, N_bins - TRIM)
    rot_speed = np.abs(rot_vel[START:END])
    activity = bin_activity[START:END]
    t_sec = bin_t_sec[START:END]
    z_angle_plt = z_angle[START:END]
    L = len(rot_speed)

    speed_c = rot_speed - rot_speed.mean()
    act_c   = activity - activity.mean()
    vel_s = box_smooth(speed_c, smooth_k)
    act_s = box_smooth(act_c, smooth_k)

    spikes_vel = find_troughs(vel_s)
    spikes_act = find_troughs(act_s)

    if len(spikes_act) > 0:
        idx_a = np.round(spikes_act).astype(int).clip(0, L - 1)
        spikes_act = spikes_act[act_s[idx_a] < np.percentile(act_s, 3)]
    if len(spikes_vel) > 0:
        idx_v = np.round(spikes_vel).astype(int).clip(0, L - 1)
        spikes_vel = spikes_vel[vel_s[idx_v] < np.percentile(vel_s, 3)]

    # Greedy matching
    offsets_bins, matched_v, matched_a = [], [], []
    if len(spikes_vel) > 0 and len(spikes_act) > 0:
        candidates = sorted([(abs(float(a) - float(b)), ia, ib)
                             for ia, a in enumerate(spikes_vel)
                             for ib, b in enumerate(spikes_act)])
        used_v, used_a = set(), set()
        for _d, ia, ib in candidates:
            if ia not in used_v and ib not in used_a:
                used_v.add(ia); used_a.add(ib)
                offsets_bins.append(float(spikes_act[ib] - spikes_vel[ia]))
                matched_v.append(float(spikes_vel[ia]))
                matched_a.append(float(spikes_act[ib]))

    offsets_bins = np.array(offsets_bins) if offsets_bins else np.array([])
    bin_dt = np.median(np.diff(t_sec)) if len(t_sec) > 1 else 1.0 / (30.0 * num_bins)
    offsets_ms = offsets_bins * bin_dt * 1000.0
    offsets_frames = offsets_bins / num_bins

    # Keep only pairs with |offset| < 100 ms
    if len(offsets_ms) > 0:
        mask = np.abs(offsets_ms) < 100.0
        offsets_bins   = offsets_bins[mask]
        offsets_ms     = offsets_ms[mask]
        offsets_frames = offsets_frames[mask]
        matched_v = [v for v, m in zip(matched_v, mask) if m]
        matched_a = [a for a, m in zip(matched_a, mask) if m]

    # Summary
    print("\n" + "=" * 60)
    print("  VOXEL-BIN TEMPORAL ALIGNMENT")
    print("=" * 60)
    print(f"  Recording: {seq_dir}")
    print(f"  Raw poses: {P}  Voxel frames: {n_voxel_frames}  Bins/frame: {num_bins}")
    print(f"  Speed troughs: {len(spikes_vel)}  Activity troughs: {len(spikes_act)}")
    if len(offsets_ms) > 0:
        for i, (sv, sa, ob, om) in enumerate(zip(matched_v, matched_a, offsets_bins, offsets_ms)):
            print(f"    pair {i}: speed@bin {sv:.0f}  activity@bin {sa:.0f}  {ob:+.1f} bins ({om:+.1f} ms)")
        print(f"  Median: {np.median(offsets_ms):+.1f} ms  ({np.median(offsets_frames):+.2f} frames)")
    print("=" * 60)

    # Figure
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1.2], hspace=0.35, wspace=0.30)
    bins_x = np.arange(L)
    color_sp, color_ev = "#2066a8", "#d6604d"

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(bins_x, np.rad2deg(z_angle_plt), color=color_sp, lw=0.8, label="Z-rotation")
    ax1.set_ylabel("Z-rot (°)", color=color_sp)
    ax1r = ax1.twinx()
    ax1r.plot(bins_x, box_smooth(activity, smooth_k), color=color_ev, lw=0.6, alpha=0.8, label="bin activity")
    ax1r.set_ylabel("Bin activity (std)", color=color_ev)
    ax1.set_title("Z-rotation & voxel-bin activity", fontsize=11); ax1.set_xlabel("Bin index")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax1r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8)

    ax2 = fig.add_subplot(gs[1, :])
    vn, an = unit_norm(vel_s), unit_norm(act_s)
    ax2.plot(bins_x, vn, color=color_sp, lw=1.0, label="Rotation speed (norm)")
    ax2.plot(bins_x, an, color=color_ev, lw=1.0, alpha=0.8, label="Activity (norm)")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    for sv in spikes_vel:
        i = int(min(round(sv), L - 1)); ax2.plot(sv, vn[i], "v", color=color_sp, ms=6, zorder=5)
    for sa in spikes_act:
        i = int(min(round(sa), L - 1)); ax2.plot(sa, an[i], "v", color=color_ev, ms=6, zorder=5)
    ax2.set_xlabel("Bin index"); ax2.set_title("Speed vs activity (▼ troughs)", fontsize=11)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[2, 0])
    if len(offsets_ms) > 0:
        ax3.hist(offsets_ms, bins=max(5, len(offsets_ms)), color="#8da0cb", edgecolor="k", linewidth=0.8, rwidth=0.85)
        ax3.axvline(0, color="red", lw=1.2, ls="--", label="zero")
        ax3.axvline(float(np.median(offsets_ms)), color="orange", lw=1.2, label=f"med={np.median(offsets_ms):+.1f}ms")
        ax3.set_xlabel("Offset (ms)"); ax3.set_ylabel("Count"); ax3.legend(fontsize=8)
    ax3.set_title("Offset distribution (ms)", fontsize=10)

    ax4 = fig.add_subplot(gs[2, 1])
    if len(offsets_ms) > 0:
        ax4.scatter(matched_v, offsets_ms, color="#333", s=30, zorder=3)
        ax4.axhline(0, color="gray", lw=0.5, ls=":")
        ax4.axhline(float(np.median(offsets_ms)), color="red", lw=1.0, ls="--")
        ax4.set_xlabel("Bin of speed trough"); ax4.set_ylabel("Offset (ms)")
        med_ms = float(np.median(offsets_ms))
        label = "✓ GOOD" if abs(med_ms) < bin_dt * 1000 else ("~ OK" if abs(med_ms) < 2 * bin_dt * 1000 else "✗ OFF")
        ax4.set_title(f"Offset over time  |  n={len(offsets_ms)}  med={med_ms:+.1f}ms  {label}", fontsize=9)
    else:
        ax4.set_title("Offset over time", fontsize=10)

    fig.suptitle("Voxel-Bin Temporal Alignment — Rotation Speed vs Activity",
                 fontsize=13, fontweight="bold", y=0.98)
    out = out_dir / "temporal_alignment_voxel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Figure saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  3. verify  (from verify_temporal_alignment.py)
# ══════════════════════════════════════════════════════════════════════

def run_verify(seq_dir: Path, out_dir: Path) -> None:
    """Verify that interpolated poses are temporally aligned with camera frames.

    Key metric: nearest_offset_ms — time distance from each frame timestamp
    to the closest raw robot-pose sample.
    """
    hdf5_dir = seq_dir / "hdf5"
    rs_path = hdf5_dir / "realsense.h5"
    poses_path = hdf5_dir / "poses.h5"
    meta_path = hdf5_dir / "metadata.h5"

    for p in (rs_path, poses_path):
        if not p.exists():
            raise FileNotFoundError(f"Required: {p}")

    with h5py.File(rs_path, "r") as f:
        t_sys_ns = f["t_sys_ns"][:]
        t_hw_ms  = f["t_hw_ms"][:]
        frame_no = f["frame_number"][:]
        # Prefer hardware-anchored timestamps (same domain as sys, but cleaned-up jitter)
        t_hw_as_sys_ns = f["t_hw_as_sys_ns"][:] if "t_hw_as_sys_ns" in f else t_sys_ns
    with h5py.File(poses_path, "r") as f:
        ee_T      = f["ee_T"][:]
        joints    = f["joint_positions"][:]
        offset_ms = f["nearest_offset_ms"][:]

    fps = float(DEFAULT_FPS); raw_poses = -1
    if meta_path.exists():
        with h5py.File(meta_path, "r") as f:
            fps = float(f.attrs.get("fps", 30.0))
            raw_poses = int(f.attrs.get("raw_poses_received", -1))

    N = len(t_sys_ns)
    t_s = (t_hw_as_sys_ns - t_hw_as_sys_ns[0]) / 1e9  # use hw-anchored times as the time axis
    # sys-vs-hw drift: how much does raw receive time deviate from hw-anchored time?
    sys_vs_hw_drift_ms = (t_sys_ns - t_hw_as_sys_ns).astype(np.float64) / 1e6
    dt_hw_ms = np.diff(t_hw_ms)
    frame_gaps = np.diff(frame_no.astype(np.int64))
    dropped = int(np.sum(frame_gaps > 1))
    total_missing = int(np.sum(frame_gaps - 1))
    ee_pos = ee_T[:, :3, 3]
    dt_for_vel = dt_hw_ms / 1000.0
    ee_speed = np.linalg.norm(np.diff(ee_pos, axis=0), axis=1) / np.where(dt_for_vel > 0, dt_for_vel, np.nan)

    target_dt_ms = 1000.0 / fps
    duration_s = t_s[-1] - t_s[0]
    actual_fps = (N - 1) / duration_s if duration_s > 0 else 0.0
    half_frame_ms = 1000.0 / (2.0 * actual_fps) if actual_fps > 0 else 16.7

    off_median = float(np.median(offset_ms))
    off_p95    = float(np.percentile(offset_ms, 95))
    off_max    = float(np.max(offset_ms))
    dt_median  = float(np.median(dt_hw_ms))
    dt_std     = float(np.std(dt_hw_ms))

    aligned = off_median < half_frame_ms and off_p95 < target_dt_ms
    verdict = "WELL-ALIGNED" if aligned else "POTENTIALLY MISALIGNED"
    verdict_color = "#1a7431" if aligned else "#b22222"

    print(f"\n{'='*56}")
    print(f"  Verdict: {verdict}")
    print(f"  Offset: median={off_median:.2f}ms  p95={off_p95:.2f}ms  max={off_max:.2f}ms")
    print(f"  ½-frame: {half_frame_ms:.2f}ms")
    print(f"{'='*56}")

    def _color_bg(ax, good):
        ax.set_facecolor("#e8f5e9" if good else "#ffebee")

    fig = plt.figure(figsize=(20, 14)); fig.patch.set_facecolor("#f5f5f5")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    fig.suptitle(f"Temporal Alignment — {seq_dir.name}\n{verdict}  |  {N} frames, "
                 f"{duration_s:.1f}s @ {actual_fps:.1f}fps  |  raw poses: {raw_poses}",
                 fontsize=13, fontweight="bold", color=verdict_color, y=0.995)

    # 1: offset over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(t_s, offset_ms, alpha=0.25, color="steelblue")
    ax1.plot(t_s, offset_ms, color="steelblue", lw=0.7, alpha=0.85)
    ax1.axhline(off_median, color="navy", ls="--", lw=1.5, label=f"median {off_median:.1f}ms")
    ax1.axhline(off_p95, color="darkorange", ls="-.", lw=1.5, label=f"p95 {off_p95:.1f}ms")
    ax1.axhline(half_frame_ms, color="red", ls=":", lw=1.8, label=f"½frame {half_frame_ms:.1f}ms")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Offset (ms)")
    ax1.set_title("Nearest raw-pose offset [PRIMARY]", fontweight="bold")
    ax1.set_ylim(bottom=0); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    _color_bg(ax1, off_median < half_frame_ms)

    # 2: offset histogram
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(offset_ms, bins=60, color="steelblue", edgecolor="white", alpha=0.80)
    ax2.axvline(off_median, color="navy", ls="--", lw=1.5)
    ax2.axvline(off_p95, color="darkorange", ls="-.", lw=1.5)
    ax2.axvline(half_frame_ms, color="red", ls=":", lw=1.8)
    ax2.set_xlabel("Offset (ms)"); ax2.set_ylabel("Count"); ax2.set_title("Offset distribution")
    ax2.grid(True, alpha=0.3)

    # 3: sys-vs-hw drift over time  (how noisy is the Python receive time vs hw clock)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(t_s[1:], sys_vs_hw_drift_ms[1:], color="teal", lw=0.7, alpha=0.80)
    ax3.axhline(float(np.median(sys_vs_hw_drift_ms[1:])), color="darkgreen", ls="--", lw=1.5,
                label=f"median {np.median(sys_vs_hw_drift_ms[1:]):.1f}ms")
    ax3.axhline(0, color="gray", ls=":", lw=1.0)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("sys − hw_as_sys (ms)")
    ax3.set_title("Sys receive-time jitter vs hardware clock [clock drift check]")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # 4: hw frame interval histogram (regularity of depth capture)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(dt_hw_ms, bins=60, color="teal", edgecolor="white", alpha=0.80)
    ax4.axvline(target_dt_ms, color="red", ls="--", lw=1.5)
    ax4.set_xlabel("Interval (ms)"); ax4.set_ylabel("Count"); ax4.grid(True, alpha=0.3)

    # 5: EE position
    ax5 = fig.add_subplot(gs[2, :2])
    for i, (label, col) in enumerate(zip(["x", "y", "z"], ["#e74c3c", "#27ae60", "#2980b9"])):
        ax5.plot(t_s, ee_pos[:, i], color=col, lw=0.9, alpha=0.85, label=label)
    ax5.set_xlabel("Time (s)"); ax5.set_ylabel("EE position (m)")
    ax5.set_title("EE XYZ trajectory"); ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

    # 6: EE speed
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(t_s[1:], ee_speed * 100, color="purple", lw=0.8, alpha=0.80)
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Speed (cm/s)")
    ax6.set_title("EE speed"); ax6.grid(True, alpha=0.3)

    out = out_dir / "verify_temporal_alignment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"Figure saved → {out}")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pose temporal alignment analysis — generates diagnostic plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Analysis to run")

    p = sub.add_parser("frame_level", help="Frame-level temporal alignment")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--smooth", type=int, default=3)

    p = sub.add_parser("voxel_level", help="Voxel-bin level temporal alignment")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--smooth", type=int, default=5)

    p = sub.add_parser("verify", help="Verify interpolated pose alignment")
    p.add_argument("--data_dir", type=str, required=True)

    p = sub.add_parser("all", help="Run all analyses")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--smooth", type=int, default=5)

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

    if cmd == "frame_level":
        run_frame_level(seq_dir, out_dir, smooth_k=args.smooth)
    elif cmd == "voxel_level":
        run_voxel_level(seq_dir, out_dir, smooth_k=args.smooth)
    elif cmd == "verify":
        run_verify(seq_dir, out_dir)
    elif cmd == "all":
        print(f"Running all pose alignment analyses for {seq_dir.name}\n")
        errors = []
        for name, fn in [
            ("frame_level",  lambda: run_frame_level(seq_dir, out_dir, smooth_k=args.smooth)),
            ("voxel_level",  lambda: run_voxel_level(seq_dir, out_dir, smooth_k=args.smooth)),
            ("verify",       lambda: run_verify(seq_dir, out_dir)),
        ]:
            try:
                print(f"\n{'='*60}\n  {name}\n{'='*60}")
                fn()
            except Exception as e:
                print(f"  SKIPPED ({name}): {e}")
                errors.append(name)
        if errors:
            print(f"\nSkipped: {', '.join(errors)}")
        print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
