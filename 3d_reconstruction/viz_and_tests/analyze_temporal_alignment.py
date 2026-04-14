#!/usr/bin/env python3
"""
Analyse temporal alignment between robot rotation reversals and event-camera
activity from a ``temporal_check`` recording.

The robot holds a fixed position and oscillates its Z-rotation.  Every direction
reversal produces a dip in event activity (the arm is momentarily still).
This script:

  1. Extracts the Z-rotation angle and rotation speed from the EE poses.
  2. Extracts the per-frame event activity from the event frames.
  3. Detects prominent spikes in both signals.
  4. Computes per-spike temporal offsets (pose vs event).
  5. Produces a multi-panel figure saved next to the recording.

Usage (standalone):
    python3 analyze_temporal_alignment.py --data_dir data/real/temporal_check

Called automatically by ``synchronised_recording.py --temporal-check``.
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
    """1-D causal box filter (kernel width *k*).  k <= 1 is a no-op."""
    if k <= 1:
        return arr.astype(np.float64).copy()
    kernel = np.ones(k) / k
    # 'same' keeps array length; edge effects are minor for our purposes
    return np.convolve(arr.astype(np.float64), kernel, mode="same")


def zero_crossings(sig: np.ndarray) -> np.ndarray:
    """Return indices where *sig* changes sign (linear interpolation)."""
    s = np.sign(sig)
    diff = np.diff(s)
    idx = np.nonzero(diff)[0]
    # Fractional index via linear interpolation between idx and idx+1
    frac = sig[idx] / (sig[idx] - sig[idx + 1])
    return idx + frac


def find_troughs(sig: np.ndarray, min_prominence_frac: float = 0.15) -> np.ndarray:
    """Find troughs in *sig* using scipy.signal.find_peaks on the inverted signal.

    Uses proper prominence (height above the lowest saddle to any higher
    neighbour), which is robust to small oscillations on the flanks of deep
    troughs.  Returns float indices.
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

def analyze_temporal_alignment(
    seq_dir: Path,
    smooth_k: int = 3,
    out_path: Path | None = None,
) -> None:
    """Run the full temporal-alignment analysis and save a figure.

    Parameters
    ----------
    seq_dir : Path
        Recording directory (e.g. ``data/real/temporal_check``).
    smooth_k : int
        Box-filter width for smoothing both signals before zero-crossing
        detection.
    out_path : Path or None
        Where to save the figure.  Defaults to ``seq_dir/temporal_alignment.png``.
    """
    if out_path is None:
        out_path = seq_dir / "temporal_alignment.png"

    hdf5_dir = seq_dir / "hdf5"

    # ── load poses ─────────────────────────────────────────────────────────
    poses_path = hdf5_dir / "poses.h5"
    if not poses_path.exists():
        raise FileNotFoundError(f"No poses.h5 in {hdf5_dir}")
    with h5py.File(poses_path, "r") as f:
        ee_T = f["ee_T"][:]               # (N, 4, 4)
        if "t_ns" in f:
            t_ns = f["t_ns"][:]
        else:
            # Fall back to frame indices
            t_ns = np.arange(len(ee_T), dtype=np.float64)

    N = len(ee_T)
    # Extract Z-rotation angle from rotation matrices.
    # The EE points down (180° about X), so the Z-rotation component is
    # encoded in the upper-left 2×2 block of R.  We use atan2(R[1,0], R[0,0])
    # which gives the rotation about the Z axis regardless of the base tilt.
    R = ee_T[:, :3, :3]                              # (N, 3, 3)
    z_angle = np.arctan2(R[:, 1, 0], R[:, 0, 0])     # radians
    # Convert timestamps to seconds from start
    t_sec = (t_ns - t_ns[0]) / 1e9 if t_ns.dtype != np.float64 or t_ns.max() > 1e6 else t_ns.copy()
    # Rotation speed (finite difference, frame-rate normalised)
    dt = np.diff(t_sec)
    dt[dt == 0] = 1.0  # avoid division by zero
    rot_vel = np.zeros(N, dtype=np.float64)
    rot_vel[1:] = np.diff(z_angle) / dt
    # Handle angle wrapping: clamp large jumps
    rot_vel[np.abs(rot_vel) > 10.0] = 0.0

    # ── load event count ───────────────────────────────────────────────
    ev_h5 = hdf5_dir / "events_cam0.h5"
    if not ev_h5.exists():
        raise FileNotFoundError(f"No events_cam0.h5 in {hdf5_dir}")
    with h5py.File(ev_h5, "r") as f:
        if "events/frames" in f:
            ev_frames = f["events/frames"][:]   # (M, H, W) uint8
        else:
            raise KeyError("No events/frames dataset in events_cam0.h5")

    # Event activity: mean absolute deviation from neutral background (128).
    # 0 when the frame is completely idle; rises with more events regardless
    # of polarity balance.  Much more robust than counting non-128 pixels.
    # Use per-frame pixel std as the activity metric.
    # This is background-agnostic: std=0 when all pixels are uniform (no events),
    # and increases as more pixels are pushed to white/black by events.
    # mean(|frame-128|) was wrong because the Metavision SDK default background
    # is ~0 (dark), not 128, so the deviation-from-128 was inversely correlated
    # with actual event activity.
    event_activity = ev_frames.astype(np.float64).std(axis=(1, 2))
    M = len(event_activity)

    # Truncate to common length, then trim first 100 and last 100 frames.
    L = min(N, M)
    START = min(100, L)
    END = max(START, L - 100)
    rot_speed = np.abs(rot_vel[START:END])     # rotation speed (rad/s)
    event_activity = event_activity[START:END]
    t_sec = t_sec[START:END]
    z_angle_plot = z_angle[START:END]
    L = len(rot_speed)

    # Mean-subtract both signals so zero-crossings mark transitions
    # between high-activity (rotating) and low-activity (reversal) regions.
    speed_centered = rot_speed - rot_speed.mean()
    evcount_centered = event_activity - event_activity.mean()

    # ── smooth & detect spikes ─────────────────────────────────────────────
    vel_smooth = box_smooth(speed_centered, smooth_k)
    ev_smooth  = box_smooth(evcount_centered, smooth_k)

    spikes_vel = find_troughs(vel_smooth)
    spikes_ev  = find_troughs(ev_smooth)

    # Keep only troughs below 20th percentile (filters noise minima while
    # retaining all genuine turnaround/activity troughs).
    if len(spikes_ev) > 0:
        idx_ev = np.round(spikes_ev).astype(int).clip(0, L - 1)
        ev_thresh = np.percentile(ev_smooth, 20)
        spikes_ev = spikes_ev[ev_smooth[idx_ev] < ev_thresh]

    if len(spikes_vel) > 0:
        idx_vel = np.round(spikes_vel).astype(int).clip(0, L - 1)
        vel_thresh = np.percentile(vel_smooth, 20)
        spikes_vel = spikes_vel[vel_smooth[idx_vel] < vel_thresh]

    # ── match nearest spikes (each event spike used at most once) ──────────
    offsets_frames: list[float] = []
    matched_vel: list[float] = []
    matched_ev: list[float] = []
    used_ev: set[int] = set()
    for sv in spikes_vel:
        if len(spikes_ev) == 0:
            break
        dists = np.abs(spikes_ev - sv)
        for best in np.argsort(dists):
            if best not in used_ev:
                used_ev.add(int(best))
                offsets_frames.append(float(spikes_ev[best] - sv))
                matched_vel.append(float(sv))
                matched_ev.append(float(spikes_ev[best]))
                break

    offsets_frames = np.array(offsets_frames) if offsets_frames else np.array([])
    int_offsets = np.round(offsets_frames).astype(int) if len(offsets_frames) > 0 else np.array([], dtype=int)
    # Keep ms conversion for reference
    if len(t_sec) > 1:
        frame_dt = np.median(np.diff(t_sec))
    else:
        frame_dt = 1.0 / 30.0
    offsets_ms = offsets_frames * frame_dt * 1000.0

    # ── print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 60)
    print(f"  Recording:       {seq_dir}")
    print(f"  Frames (pose):   {N}")
    print(f"  Frames (event):  {M}")
    print(f"  Common length:   {L}")
    print(f"  Frame interval:  {frame_dt*1000:.1f} ms  ({1/frame_dt:.1f} Hz)")
    print(f"  Smoothing:       k={smooth_k}")
    print()
    print(f"  Speed spikes detected:         {len(spikes_vel)}")
    print(f"  Event-activity spikes detected: {len(spikes_ev)}")
    if len(int_offsets) > 0:
        print(f"  Per-spike frame offsets  [event − speed]:")
        for i, (ov, oe, off_fr, off_ms) in enumerate(
                zip(matched_vel, matched_ev, int_offsets, offsets_ms)):
            print(f"    pair {i}: speed@{ov:.0f}  event@{oe:.0f}  "
                  f"offset = {off_fr:+d} frames  ({off_ms:+.1f} ms)")
        print(f"  Mean offset:     {int_offsets.mean():+.1f} frames  "
              f"({offsets_ms.mean():+.1f} ms)")
        print(f"  Median offset:   {np.median(int_offsets):+.1f} frames  "
              f"({np.median(offsets_ms):+.1f} ms)")
        print(f"  Std offset:      {int_offsets.std():.1f} frames")
    else:
        print("  No matched spikes found.")
    print("=" * 60)

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 1.2, 1.2],
                  hspace=0.35, wspace=0.30)

    frames = np.arange(L)

    # --- Panel 1: Z-rotation angle + event activity (dual y-axis) ---
    ax1 = fig.add_subplot(gs[0, :])
    color_pos = "#2066a8"
    color_ev = "#d6604d"
    ax1.plot(frames, np.rad2deg(z_angle_plot), color=color_pos, lw=1.4, label="EE Z-rotation")
    ax1.set_ylabel("Z-rotation (°)", color=color_pos)
    ax1.tick_params(axis="y", labelcolor=color_pos)
    ax1r = ax1.twinx()
    ev_smooth_raw = box_smooth(event_activity, smooth_k)
    ax1r.plot(frames, ev_smooth_raw, color=color_ev, lw=1.0, alpha=0.8,
              label="event activity")
    ax1r.set_ylabel("Event activity (pixel std, smoothed)", color=color_ev)
    ax1r.tick_params(axis="y", labelcolor=color_ev)
    ax1.set_title("EE Z-rotation & event activity over time", fontsize=11)
    ax1.set_xlabel("Frame index")
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # --- Panel 2: Rotation speed + event activity (normalised overlay) ---
    ax2 = fig.add_subplot(gs[1, :])
    # Normalise to [-1, 1] for overlay
    def unit_norm(a):
        mx = np.abs(a).max()
        return a / mx if mx > 0 else a

    vn = unit_norm(vel_smooth)
    en = unit_norm(ev_smooth)
    ax2.plot(frames, vn, color=color_pos, lw=1.3, label="Rotation speed (normalised, mean-sub)")
    ax2.plot(frames, en, color=color_ev, lw=1.3, alpha=0.8,
             label="Event activity (normalised, mean-sub)")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    # Mark detected spikes: both speed and event troughs (▼)
    for sv in spikes_vel:
        i = int(min(round(sv), L - 1))
        ax2.plot(sv, vn[i], "v", color=color_pos, ms=7, zorder=5)
    for se in spikes_ev:
        i = int(min(round(se), L - 1))
        ax2.plot(se, en[i], "v", color=color_ev, ms=7, zorder=5)
    ax2.set_ylabel("Normalised amplitude")
    ax2.set_xlabel("Frame index")
    ax2.set_title("Rotation speed vs event activity  (▲▼ = detected spikes)",
                   fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")

    # --- Panel 3: frame offset histogram ---
    ax3 = fig.add_subplot(gs[2, 0])
    if len(int_offsets) > 0:
        lo, hi = int_offsets.min(), int_offsets.max()
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)  # one bin per integer offset
        ax3.hist(int_offsets, bins=bins, color="#8da0cb", edgecolor="k",
                 linewidth=0.8, rwidth=0.85)
        ax3.axvline(0, color="red", lw=1.2, ls="--", label="zero")
        ax3.axvline(float(np.median(int_offsets)), color="orange", lw=1.2, ls="-",
                    label=f"median = {np.median(int_offsets):+.0f} fr")
        ax3.set_xlabel("Frame offset  [event − speed spike]")
        ax3.set_ylabel("Count")
        ax3.set_title("Frame-offset distribution", fontsize=10)
        ax3.set_xticks(range(lo, hi + 1))
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No reversals detected", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="red")
        ax3.set_title("Frame-offset distribution", fontsize=10)

    # --- Panel 4: frame offset over time (per matched spike pair) ---
    ax4 = fig.add_subplot(gs[2, 1])
    if len(int_offsets) > 0:
        mv_frames = np.array(matched_vel)
        ax4.scatter(mv_frames, int_offsets, color="#333333", s=35, zorder=3)
        ax4.axhline(0, color="gray", lw=0.5, ls=":")
        ax4.axhline(float(np.median(int_offsets)), color="red", lw=1.0, ls="--",
                    label=f"median = {np.median(int_offsets):+.0f} fr")
        ax4.set_xlabel("Frame index of speed spike")
        ax4.set_ylabel("Frame offset  [event − speed]")
        ax4.set_title("Frame offset over time", fontsize=10)
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, "No matched spikes", ha="center", va="center",
                 transform=ax4.transAxes, color="gray")
        ax4.set_title("Frame offset over time", fontsize=10)

    # --- Panel 5: per-pair bar chart (frame offsets) ---
    ax5 = fig.add_subplot(gs[3, 0])
    if len(int_offsets) > 0:
        bar_colors = ["#4daf4a" if o == 0 else ("#2066a8" if abs(o) <= 1 else "#ff7f00")
                      for o in int_offsets]
        ax5.bar(range(len(int_offsets)), int_offsets, color=bar_colors,
                edgecolor="k", linewidth=0.5)
        ax5.axhline(0, color="gray", lw=0.7, ls="--")
        ax5.set_xlabel("Matched pair index")
        ax5.set_ylabel("Frame offset  [event − speed]")
        ax5.set_title("Per-pair frame offset", fontsize=10)
        # colour legend
        from matplotlib.patches import Patch
        ax5.legend(handles=[
            Patch(facecolor="#4daf4a", label="0 frames"),
            Patch(facecolor="#2066a8", label="±1 frame"),
            Patch(facecolor="#ff7f00", label=">1 frame"),
        ], fontsize=7, loc="upper right")
    else:
        ax5.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax5.transAxes, color="gray")
        ax5.set_title("Per-pair frame offset", fontsize=10)

    # --- Panel 6: summary text ---
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")
    summary = (
        f"Recording: {seq_dir.name}\n"
        f"Pose frames: {N}    Event frames: {M}\n"
        f"Frame rate: {1/frame_dt:.1f} Hz    "
        f"Frame Δt: {frame_dt*1000:.1f} ms\n"
        f"Smoothing: k={smooth_k}\n\n"
    )
    if len(int_offsets) > 0:
        med_fr = int(np.median(int_offsets))
        summary += (
            f"Speed spikes:          {len(spikes_vel)}\n"
            f"Event-activity spikes: {len(spikes_ev)}\n"
            f"Matched pairs:         {len(int_offsets)}\n\n"
            f"Mean offset:   {int_offsets.mean():+.1f} fr  ({offsets_ms.mean():+.1f} ms)\n"
            f"Median offset: {med_fr:+d} fr  ({np.median(offsets_ms):+.1f} ms)\n"
            f"Std:           {int_offsets.std():.1f} frames\n"
        )
        if abs(med_fr) == 0:
            summary += "\n✓ Alignment GOOD (median = 0 frames)"
        elif abs(med_fr) == 1:
            summary += "\n~ Alignment OK (median = ±1 frame)"
        else:
            summary += f"\n✗ Alignment OFF (median = {med_fr:+d} frames)"
    else:
        summary += "No reversals matched — check recording."

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=9, family="monospace", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                       edgecolor="#cccccc"))

    fig.suptitle("Temporal Alignment Check  —  Rotation Speed vs Event Activity",
                 fontsize=13, fontweight="bold", y=0.98)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Temporal] Figure saved → {out_path}")
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse temporal alignment from a temporal_check recording",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to recording directory (e.g. data/real/temporal_check)")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Box-filter width for smoothing")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (default: <data_dir>/temporal_alignment.png)")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else None
    analyze_temporal_alignment(seq_dir, smooth_k=args.smooth, out_path=out)


if __name__ == "__main__":
    main()
