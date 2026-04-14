#!/usr/bin/env python3
"""
Voxel-bin level temporal alignment analysis.

Uses precomputed voxel grids (5 temporal bins per frame) and raw poses
(~100 Hz) to measure temporal alignment at ~5× higher resolution than
the frame-level analysis.

For each voxel bin we compute:
  - a precise timestamp by linearly interpolating between the frame's
    t_ev_start_us and t_ev_end_us
  - the nearest raw pose (via elapsed-time matching, same as
    align_event_frames_to_depth)
  - rotation speed from the raw poses
  - event activity as the per-bin pixel std

Usage:
    python3 analyze_temporal_alignment_voxel.py --data_dir data/real/temporal_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── helpers ────────────────────────────────────────────────────────────────────

def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.astype(np.float64).copy()
    kernel = np.ones(k) / k
    return np.convolve(arr.astype(np.float64), kernel, mode="same")


def find_troughs(sig: np.ndarray, min_prominence_frac: float = 0.15) -> np.ndarray:
    """Find troughs using scipy.signal.find_peaks on the inverted signal.

    Uses proper prominence (height above the lowest saddle to any higher point),
    which is robust to small oscillations on the flanks of deep troughs.
    """
    if len(sig) < 3:
        return np.array([], dtype=np.float64)
    sig_range = float(sig.max() - sig.min())
    if sig_range == 0:
        return np.array([], dtype=np.float64)
    prominence_threshold = min_prominence_frac * sig_range
    peaks, _ = find_peaks(-sig, prominence=prominence_threshold)
    return peaks.astype(np.float64)


# ── main analysis ──────────────────────────────────────────────────────────────

def analyze_temporal_alignment_voxel(
    seq_dir: Path,
    smooth_k: int = 5,
    out_path: Path | None = None,
) -> None:
    if out_path is None:
        out_path = seq_dir / "temporal_alignment_voxel.png"

    hdf5_dir   = seq_dir / "hdf5"
    voxels_dir = seq_dir / "events" / "voxels_cam0"

    # ── load raw poses ─────────────────────────────────────────────────
    raw_poses_path = hdf5_dir / "raw_poses.h5"
    if not raw_poses_path.exists():
        raise FileNotFoundError(f"No raw_poses.h5 in {hdf5_dir}")
    with h5py.File(raw_poses_path, "r") as f:
        pose_t_ns = f["t_ns"][:]       # (P,) int64, system-clock ns
        pose_ee_T = f["ee_T"][:]       # (P, 4, 4) float64
    P = len(pose_t_ns)

    # ── load metadata for time conversion ──────────────────────────────
    meta_path = hdf5_dir / "metadata.h5"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.h5 in {hdf5_dir}")
    with h5py.File(meta_path, "r") as f:
        event_logging_start_ns = int(f.attrs["event_logging_start_ns"])

    # ── load event frame timestamps ────────────────────────────────────
    ev_h5 = hdf5_dir / "events_cam0.h5"
    if not ev_h5.exists():
        raise FileNotFoundError(f"No events_cam0.h5 in {hdf5_dir}")
    with h5py.File(ev_h5, "r") as f:
        t_start_us = f["events/t_ev_start_us"][:]  # (N,) µs event-clock
        t_end_us   = f["events/t_ev_end_us"][:]    # (N,) µs event-clock
    N_frames = len(t_start_us)

    # ── load voxel grids and compute per-bin activity + timestamps ─────
    if not voxels_dir.exists():
        raise FileNotFoundError(f"No voxels directory: {voxels_dir}")
    voxel_files = sorted(voxels_dir.glob("voxel_*.npy"))
    if len(voxel_files) == 0:
        raise FileNotFoundError(f"No voxel_*.npy files in {voxels_dir}")

    n_voxel_frames = min(len(voxel_files), N_frames)
    sample = np.load(voxel_files[0])
    num_bins = sample.shape[0]

    # Per-bin timestamps (µs, event-camera clock) and activity
    bin_t_us   = []   # µs in event-camera clock
    bin_activity = []

    for frame_idx in range(n_voxel_frames):
        voxel = np.load(voxels_dir / f"voxel_{frame_idx:06d}.npy")  # (B, H, W)
        t0 = float(t_start_us[frame_idx])
        t1 = float(t_end_us[frame_idx])
        for b in range(num_bins):
            # Each bin spans a sub-interval: bin centres
            frac = (b + 0.5) / num_bins
            bin_t_us.append(t0 + frac * (t1 - t0))
            # Activity = std of the bin's spatial map
            bin_activity.append(float(np.std(voxel[b])))

    bin_t_us = np.array(bin_t_us)
    bin_activity = np.array(bin_activity)
    N_bins = len(bin_t_us)

    # ── convert to elapsed ns (common reference) ──────────────────────
    # Event camera: elapsed from first event frame's end timestamp
    ev_elapsed_ns = (bin_t_us - t_end_us[0]).astype(np.float64) * 1000.0  # µs → ns

    # Poses: elapsed from event_logging_start_ns (same reference as
    # align_event_frames_to_depth)
    pose_elapsed_ns = (pose_t_ns - event_logging_start_ns).astype(np.float64)

    # ── assign nearest raw pose to each voxel bin ─────────────────────
    indices = np.searchsorted(pose_elapsed_ns, ev_elapsed_ns, side="left")
    indices = np.clip(indices, 0, P - 1)
    # Check if left neighbour is closer
    left = np.clip(indices - 1, 0, P - 1)
    d_right = np.abs(pose_elapsed_ns[indices] - ev_elapsed_ns)
    d_left  = np.abs(pose_elapsed_ns[left] - ev_elapsed_ns)
    use_left = d_left < d_right
    indices[use_left] = left[use_left]

    # Matched poses → (N_bins, 4, 4)
    matched_ee_T = pose_ee_T[indices]

    # ── compute rotation speed from matched poses ─────────────────────
    R = matched_ee_T[:, :3, :3]
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    # Time in seconds for each bin
    bin_t_sec = (ev_elapsed_ns - ev_elapsed_ns[0]) / 1e9
    dt = np.diff(bin_t_sec)
    dt[dt == 0] = 1e-6
    rot_vel = np.zeros(N_bins, dtype=np.float64)
    rot_vel[1:] = np.diff(z_angle) / dt
    rot_vel[np.abs(rot_vel) > 10.0] = 0.0  # clamp wrapping artefacts

    # ── trim start and end ─────────────────────────────────────────────
    TRIM = 100 * num_bins  # 100 frames worth of bins
    START = min(TRIM, N_bins)
    END   = max(START, N_bins - TRIM)
    rot_speed   = np.abs(rot_vel[START:END])
    activity    = bin_activity[START:END]
    t_sec       = bin_t_sec[START:END]
    z_angle_plt = z_angle[START:END]
    L = len(rot_speed)

    # ── normalise & detect spikes ─────────────────────────────────────
    speed_c  = rot_speed - rot_speed.mean()
    act_c    = activity  - activity.mean()

    vel_s  = box_smooth(speed_c, smooth_k)
    act_s  = box_smooth(act_c, smooth_k)

    spikes_vel = find_troughs(vel_s)
    spikes_act = find_troughs(act_s)

    # Keep only troughs below 3rd percentile (filters high-frequency noise
    # minima while retaining all genuine turnaround/activity troughs)
    if len(spikes_act) > 0:
        idx_a = np.round(spikes_act).astype(int).clip(0, L - 1)
        thr_a = np.percentile(act_s, 3)
        spikes_act = spikes_act[act_s[idx_a] < thr_a]
    if len(spikes_vel) > 0:
        idx_v = np.round(spikes_vel).astype(int).clip(0, L - 1)
        thr_v = np.percentile(vel_s, 3)
        spikes_vel = spikes_vel[vel_s[idx_v] < thr_v]

    # ── greedy symmetric nearest-neighbour matching ───────────────────
    offsets_bins: list[float] = []
    matched_v: list[float] = []
    matched_a: list[float] = []

    if len(spikes_vel) > 0 and len(spikes_act) > 0:
        candidates = sorted(
            [(abs(float(a) - float(b)), ia, ib)
             for ia, a in enumerate(spikes_vel)
             for ib, b in enumerate(spikes_act)]
        )
        used_v: set[int] = set()
        used_a: set[int] = set()
        for _dist, ia, ib in candidates:
            if ia not in used_v and ib not in used_a:
                used_v.add(ia)
                used_a.add(ib)
                offsets_bins.append(float(spikes_act[ib] - spikes_vel[ia]))
                matched_v.append(float(spikes_vel[ia]))
                matched_a.append(float(spikes_act[ib]))

    offsets_bins = np.array(offsets_bins) if offsets_bins else np.array([])

    # Convert bin offsets to ms
    if len(t_sec) > 1:
        bin_dt = np.median(np.diff(t_sec))
    else:
        bin_dt = 1.0 / (30.0 * num_bins)
    offsets_ms = offsets_bins * bin_dt * 1000.0
    # Also express as frame offsets
    offsets_frames = offsets_bins / num_bins

    # ── print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VOXEL-BIN TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 60)
    print(f"  Recording:        {seq_dir}")
    print(f"  Raw poses:        {P}")
    print(f"  Voxel frames:     {n_voxel_frames}")
    print(f"  Bins per frame:   {num_bins}")
    print(f"  Total bins:       {N_bins}  (trimmed to {L})")
    print(f"  Bin interval:     {bin_dt*1000:.2f} ms  ({1/bin_dt:.0f} Hz)")
    print(f"  Smoothing:        k={smooth_k}")
    print()
    print(f"  Speed troughs:    {len(spikes_vel)}")
    print(f"  Activity troughs: {len(spikes_act)}")
    if len(offsets_ms) > 0:
        print(f"  Per-pair offsets:")
        for i, (sv, sa, ob, om) in enumerate(
                zip(matched_v, matched_a, offsets_bins, offsets_ms)):
            print(f"    pair {i}: speed@bin {sv:.0f}  activity@bin {sa:.0f}  "
                  f"offset = {ob:+.1f} bins  ({om:+.1f} ms)")
        med_ms = float(np.median(offsets_ms))
        med_fr = float(np.median(offsets_frames))
        print(f"  Mean offset:   {offsets_ms.mean():+.1f} ms  "
              f"({np.mean(offsets_frames):+.2f} frames)")
        print(f"  Median offset: {med_ms:+.1f} ms  ({med_fr:+.2f} frames)")
        print(f"  Std:           {offsets_ms.std():.1f} ms")
    else:
        print("  No matched pairs found.")
    print("=" * 60)

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 1.2, 1.2],
                  hspace=0.35, wspace=0.30)
    bins_x = np.arange(L)
    color_sp = "#2066a8"
    color_ev = "#d6604d"

    # --- Panel 1: Z-rotation angle + voxel activity (dual y) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(bins_x, np.rad2deg(z_angle_plt), color=color_sp, lw=0.8,
             label="EE Z-rotation")
    ax1.set_ylabel("Z-rotation (°)", color=color_sp)
    ax1.tick_params(axis="y", labelcolor=color_sp)
    ax1r = ax1.twinx()
    act_sm = box_smooth(activity, smooth_k)
    ax1r.plot(bins_x, act_sm, color=color_ev, lw=0.6, alpha=0.8,
              label="voxel-bin activity")
    ax1r.set_ylabel("Voxel-bin activity (pixel std, smoothed)", color=color_ev)
    ax1r.tick_params(axis="y", labelcolor=color_ev)
    ax1.set_title("EE Z-rotation & voxel-bin activity", fontsize=11)
    ax1.set_xlabel("Bin index")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # --- Panel 2: normalised overlay + detected troughs ---
    ax2 = fig.add_subplot(gs[1, :])

    def unit_norm(a):
        mx = np.abs(a).max()
        return a / mx if mx > 0 else a

    vn = unit_norm(vel_s)
    an = unit_norm(act_s)
    ax2.plot(bins_x, vn, color=color_sp, lw=1.0,
             label="Rotation speed (norm, mean-sub)")
    ax2.plot(bins_x, an, color=color_ev, lw=1.0, alpha=0.8,
             label="Voxel activity (norm, mean-sub)")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    for sv in spikes_vel:
        i = int(min(round(sv), L - 1))
        ax2.plot(sv, vn[i], "v", color=color_sp, ms=6, zorder=5)
    for sa in spikes_act:
        i = int(min(round(sa), L - 1))
        ax2.plot(sa, an[i], "v", color=color_ev, ms=6, zorder=5)
    ax2.set_ylabel("Normalised amplitude")
    ax2.set_xlabel("Bin index")
    ax2.set_title("Rotation speed vs voxel activity  (▼ = detected troughs)",
                  fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")

    # --- Panel 3: ms-offset histogram ---
    ax3 = fig.add_subplot(gs[2, 0])
    if len(offsets_ms) > 0:
        nbins = max(5, len(offsets_ms))
        ax3.hist(offsets_ms, bins=nbins, color="#8da0cb", edgecolor="k",
                 linewidth=0.8, rwidth=0.85)
        ax3.axvline(0, color="red", lw=1.2, ls="--", label="zero")
        ax3.axvline(float(np.median(offsets_ms)), color="orange", lw=1.2,
                    ls="-", label=f"median = {np.median(offsets_ms):+.1f} ms")
        ax3.set_xlabel("Offset (ms)  [activity − speed trough]")
        ax3.set_ylabel("Count")
        ax3.set_title("Offset distribution (ms)", fontsize=10)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No pairs", ha="center", va="center",
                 transform=ax3.transAxes, color="red")
        ax3.set_title("Offset distribution (ms)", fontsize=10)

    # --- Panel 4: offset over time ---
    ax4 = fig.add_subplot(gs[2, 1])
    if len(offsets_ms) > 0:
        mv = np.array(matched_v)
        ax4.scatter(mv, offsets_ms, color="#333333", s=30, zorder=3)
        ax4.axhline(0, color="gray", lw=0.5, ls=":")
        ax4.axhline(float(np.median(offsets_ms)), color="red", lw=1.0,
                    ls="--", label=f"median = {np.median(offsets_ms):+.1f} ms")
        ax4.set_xlabel("Bin index of speed trough")
        ax4.set_ylabel("Offset (ms)")
        ax4.set_title("Offset over time", fontsize=10)
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No pairs", ha="center", va="center",
                 transform=ax4.transAxes, color="gray")
        ax4.set_title("Offset over time", fontsize=10)

    # --- Panel 5: per-pair bar ---
    ax5 = fig.add_subplot(gs[3, 0])
    if len(offsets_ms) > 0:
        bar_colors = ["#4daf4a" if abs(o) < bin_dt * 1000 else
                      ("#2066a8" if abs(o) < 2 * bin_dt * 1000 else "#ff7f00")
                      for o in offsets_ms]
        ax5.bar(range(len(offsets_ms)), offsets_ms, color=bar_colors,
                edgecolor="k", linewidth=0.5)
        ax5.axhline(0, color="gray", lw=0.7, ls="--")
        ax5.set_xlabel("Matched pair index")
        ax5.set_ylabel("Offset (ms)")
        ax5.set_title("Per-pair offset", fontsize=10)
        from matplotlib.patches import Patch
        ax5.legend(handles=[
            Patch(facecolor="#4daf4a", label="< 1 bin"),
            Patch(facecolor="#2066a8", label="1–2 bins"),
            Patch(facecolor="#ff7f00", label="> 2 bins"),
        ], fontsize=7, loc="upper right")
    else:
        ax5.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax5.transAxes, color="gray")
        ax5.set_title("Per-pair offset", fontsize=10)

    # --- Panel 6: summary ---
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")
    summary = (
        f"Recording: {seq_dir.name}\n"
        f"Raw poses: {P}    Voxel frames: {n_voxel_frames}\n"
        f"Bins/frame: {num_bins}    Total bins: {N_bins} → {L}\n"
        f"Bin interval: {bin_dt*1000:.2f} ms  ({1/bin_dt:.0f} Hz)\n"
        f"Smoothing: k={smooth_k}\n\n"
    )
    if len(offsets_ms) > 0:
        med_ms = float(np.median(offsets_ms))
        med_fr = float(np.median(offsets_frames))
        summary += (
            f"Speed troughs:    {len(spikes_vel)}\n"
            f"Activity troughs: {len(spikes_act)}\n"
            f"Matched pairs:    {len(offsets_ms)}\n\n"
            f"Mean offset:   {offsets_ms.mean():+.1f} ms  "
            f"({np.mean(offsets_frames):+.2f} fr)\n"
            f"Median offset: {med_ms:+.1f} ms  ({med_fr:+.2f} fr)\n"
            f"Std:           {offsets_ms.std():.1f} ms\n"
        )
        if abs(med_ms) < bin_dt * 1000:
            summary += "\n✓ Alignment GOOD (< 1 voxel bin)"
        elif abs(med_ms) < 2 * bin_dt * 1000:
            summary += "\n~ Alignment OK (1–2 bins)"
        else:
            summary += "\n✗ Alignment may be OFF (> 2 bins)"
    else:
        summary += "No troughs matched — check recording."

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                       edgecolor="#cccccc"))

    fig.suptitle("Voxel-Bin Temporal Alignment  —  Rotation Speed vs Voxel Activity",
                 fontsize=13, fontweight="bold", y=0.98)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Temporal-Voxel] Figure saved → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Voxel-bin level temporal alignment analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Recording directory (e.g. data/real/temporal_check)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Box-filter smoothing width")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else None
    analyze_temporal_alignment_voxel(seq_dir, smooth_k=args.smooth, out_path=out_path)


if __name__ == "__main__":
    main()
