#!/usr/bin/env python3
"""
Light-check temporal alignment analysis.

Compares per-frame RGB brightness change (RealSense) against per-frame event
activity (event camera) to verify that the two streams are synchronised after
alignment.  Point both cameras at a screen that periodically switches between
black and white: each switch produces a large brightness delta in RGB and a
burst of events simultaneously.  If the two signals peak at the same frame
index the alignment is correct.

Subcommands:
  align     – Overlay RGB-delta and event-activity curves on a single plot;
              also shows the per-frame alignment_offset_ms stored during
              align_event_frames_to_depth.
  peaks     – Detect blink peaks in both signals and plot the per-peak delay.
  voxels    – Plot per-bin event activity at native bin timestamps against
              per-frame RGB delta, detect peaks in both, and match each
              event peak to the nearest RGB peak that is immediately
              followed by a significant activity drop (screen turns off).
  all       – Run all three analyses.

Usage:
    python light_check_analysis.py align  --data_dir data/light_check
    python light_check_analysis.py peaks  --data_dir data/light_check
    python light_check_analysis.py voxels --data_dir data/light_check
    python light_check_analysis.py all    --data_dir data/light_check
    python light_check_analysis.py all    --data_dir data/light_check --cam 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction_config import FPS as DEFAULT_FPS


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def ensure_out_dir(data_dir: Path) -> Path:
    d = data_dir / "light_check_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize(arr: np.ndarray) -> np.ndarray:
    """Scale arr to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr, dtype=np.float64)


def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.astype(np.float64).copy()
    return np.convolve(arr.astype(np.float64), np.ones(k) / k, mode="same")


# ══════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════

def load_rgb_delta(hdf5_dir: Path) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load per-frame RGB brightness change from realsense.h5.

    Returns
    -------
    t_depth_ms : (N,) float64  — t_global_ms (depth sensor timestamps)
    delta      : (N,) float64  — mean absolute frame-to-frame pixel difference
                                 (index 0 is NaN; index 1 onward is valid)
    t_rgb_ms   : (N,) float64 or None — color sensor timestamps if present
    """
    rs_h5 = hdf5_dir / "realsense.h5"
    if not rs_h5.exists():
        raise FileNotFoundError(f"realsense.h5 not found in {hdf5_dir}")

    with h5py.File(rs_h5, "r") as f:
        rgb = f["rgb"][:]                # (N, H, W, 3) uint8
        t_depth_ms = f["t_global_ms"][:] # (N,) float64
        t_rgb_ms = f["t_rgb_ms"][:] if "t_rgb_ms" in f else None

    # Convert to float grayscale and compute frame-to-frame absolute difference
    gray = rgb.mean(axis=-1).astype(np.float32)   # (N, H, W)
    delta = np.empty(len(gray), dtype=np.float64)
    delta[0] = np.nan
    for i in range(1, len(gray)):
        delta[i] = np.abs(gray[i].astype(np.float32) - gray[i - 1].astype(np.float32)).mean()

    return t_depth_ms, delta, t_rgb_ms


def load_event_activity(hdf5_dir: Path, cam_idx: int = 0) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load per-frame event activity and alignment offset from events_cam{i}.h5.

    Event activity = mean of absolute pixel values in the event frame
    (bright pixels = many events; dark pixels = few/no events).

    Returns
    -------
    activity : (N,) float64
    offset_ms : (N,) float64 or None  — alignment_offset_ms if present
    """
    ev_h5 = hdf5_dir / f"events_cam{cam_idx}.h5"
    if not ev_h5.exists():
        raise FileNotFoundError(f"events_cam{cam_idx}.h5 not found in {hdf5_dir}")

    with h5py.File(ev_h5, "r") as f:
        frames = f["events/frames"][:]   # (N, H, W) uint8
        offset_ms = f["events/alignment_offset_ms"][:] if "events/alignment_offset_ms" in f else None

    activity = frames.astype(np.float64).mean(axis=(1, 2))
    return activity, offset_ms


# ══════════════════════════════════════════════════════════════════════
#  1. align  — overlay both signals
# ══════════════════════════════════════════════════════════════════════

def run_align(data_dir: Path, cam_idx: int = 0, smooth_k: int = 3) -> None:
    """
    Multi-panel plot:
      1 — normalised RGB Δ and event activity vs frame index
      2 — same signals vs wall-clock time (ms, using color sensor timestamps)
      3 — per-frame alignment_offset_ms (event-to-depth, from align step)
      4 — depth-vs-RGB inter-sensor timestamp offset within each frameset
          (only shown when t_rgb_ms is present in realsense.h5)
      5 — elapsed time since start for depth frames and event bins plotted
          together; divergence reveals clock-rate differences between the
          depth sensor and the event camera oscillator
    """
    hdf5_dir = data_dir / "hdf5"
    out_dir = ensure_out_dir(data_dir)

    t_depth_ms, rgb_delta, t_rgb_ms = load_rgb_delta(hdf5_dir)
    event_activity, offset_ms = load_event_activity(hdf5_dir, cam_idx)

    N = min(len(t_depth_ms), len(event_activity))
    t_depth_ms = t_depth_ms[:N]
    rgb_delta = rgb_delta[:N]
    event_activity = event_activity[:N]

    # Load raw event bin timestamps for the elapsed-time comparison panel
    ev_elapsed_ms: Optional[np.ndarray] = None
    ev_h5_path = hdf5_dir / f"events_cam{cam_idx}.h5"
    if ev_h5_path.exists():
        with h5py.File(ev_h5_path, "r") as _f:
            if "events/t_ev_start_us" in _f and "events/t_ev_end_us" in _f:
                _t_start = _f["events/t_ev_start_us"][:N]
                _t_end   = _f["events/t_ev_end_us"][:N]
                _center  = (_t_start.astype(np.int64) + _t_end.astype(np.int64)) // 2
                ev_elapsed_ms = (_center - _center[0]) / 1000.0  # µs → ms

    # Use RGB timestamps for wall-clock axis when available; fall back to depth.
    if t_rgb_ms is not None:
        t_rgb_ms = t_rgb_ms[:N]
        t_wall_ms = t_rgb_ms
        wall_label = "Elapsed time (ms, from first RGB frame  [t_rgb_ms])"
    else:
        t_wall_ms = t_depth_ms
        wall_label = "Elapsed time (ms, from first depth frame  [t_global_ms])"
    t_rel_ms = t_wall_ms - t_wall_ms[0]

    frames = np.arange(N)

    # Smooth before normalizing so smoothing doesn't distort the scale
    rgb_s = normalize(box_smooth(np.nan_to_num(rgb_delta), smooth_k))
    ev_s = normalize(box_smooth(event_activity, smooth_k))

    has_align_offset = offset_ms is not None
    has_sensor_offset = t_rgb_ms is not None
    has_elapsed_cmp   = ev_elapsed_ms is not None
    n_panels = 2 + int(has_align_offset) + int(has_sensor_offset) + int(has_elapsed_cmp)
    fig = plt.figure(figsize=(16, 4 * n_panels))
    gs = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.45)
    panel = 0

    # ── Panel 1: vs frame index ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[panel]); panel += 1
    ax1.plot(frames, rgb_s, color="#2196F3", lw=1.2, label="RGB Δ (normalised)")
    ax1.plot(frames, ev_s, color="#FF5722", lw=1.2, alpha=0.85,
             label=f"Event activity cam{cam_idx} (normalised)")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Normalised signal")
    ax1.set_title("RGB brightness change vs event activity — by frame index")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0, N - 1)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: vs wall-clock time (RGB timestamps) ─────────────────
    ax2 = fig.add_subplot(gs[panel]); panel += 1
    ax2.plot(t_rel_ms, rgb_s, color="#2196F3", lw=1.2, label="RGB Δ")
    ax2.plot(t_rel_ms, ev_s, color="#FF5722", lw=1.2, alpha=0.85,
             label=f"Event activity cam{cam_idx}")
    ax2.set_xlabel(wall_label)
    ax2.set_ylabel("Normalised signal")
    ax2.set_title("RGB brightness change vs event activity — by wall-clock time")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(t_rel_ms[0], t_rel_ms[-1])
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: event-to-depth alignment offset ─────────────────────
    if has_align_offset:
        off = offset_ms[:N]
        ax3 = fig.add_subplot(gs[panel]); panel += 1
        ax3.plot(frames, off, color="#9C27B0", lw=1.0)
        ax3.axhline(0, color="black", lw=0.8, ls="--")
        med = np.median(off)
        ax3.axhline(med, color="#9C27B0", lw=1.2, ls="--",
                    label=f"median = {med:.2f} ms")
        ax3.fill_between(frames, off, 0, where=(off > 0), alpha=0.2,
                          color="#9C27B0", label="ev ahead of depth")
        ax3.fill_between(frames, off, 0, where=(off < 0), alpha=0.2,
                          color="#FF9800", label="ev behind depth")
        ax3.set_xlabel("Frame index")
        ax3.set_ylabel("Alignment offset (ms)")
        ax3.set_title(f"Per-frame event-to-depth alignment offset (cam{cam_idx})")
        ax3.legend(loc="upper right", fontsize=9)
        ax3.set_xlim(0, N - 1)
        ax3.grid(True, alpha=0.3)

    # ── Panel 4: depth-vs-RGB inter-sensor offset within frameset ────
    if has_sensor_offset:
        sensor_off = t_rgb_ms - t_depth_ms   # ms; positive = color later than depth
        ax4 = fig.add_subplot(gs[panel]); panel += 1
        ax4.plot(frames, sensor_off, color="#00897B", lw=1.0)
        ax4.axhline(0, color="black", lw=0.8, ls="--")
        s_med = np.median(sensor_off)
        ax4.axhline(s_med, color="#00897B", lw=1.2, ls="--",
                    label=f"median = {s_med:+.2f} ms")
        ax4.set_xlabel("Frame index")
        ax4.set_ylabel("t_rgb_ms − t_global_ms  (ms)")
        ax4.set_title(
            "Depth-vs-RGB inter-sensor timestamp offset within each RealSense frameset\n"
            "(≈ 0 ms means sensors are perfectly synchronised)"
        )
        ax4.legend(loc="upper right", fontsize=9)
        ax4.set_xlim(0, N - 1)
        ax4.grid(True, alpha=0.3)
        print(
            f"[LightCheck] Depth-vs-RGB sensor offset: "
            f"median {s_med:+.2f} ms, "
            f"max abs {np.max(np.abs(sensor_off)):.2f} ms"
        )

    # ── Panel 5: elapsed-time comparison depth vs event ─────────────
    if has_elapsed_cmp:
        depth_elapsed = t_depth_ms - t_depth_ms[0]  # ms from first depth frame
        ax5 = fig.add_subplot(gs[panel]); panel += 1
        ax5.plot(frames, depth_elapsed,  color="#1565C0", lw=1.2,
                 label="Depth elapsed  (t_global_ms − t_global_ms[0])")
        ax5.plot(frames, ev_elapsed_ms, color="#E64A19", lw=1.2, alpha=0.85,
                 label=f"Event cam{cam_idx} elapsed  (bin-center µs → ms)")
        ax5.set_xlabel("Frame index")
        ax5.set_ylabel("Elapsed time since start (ms)")
        ax5.set_title(
            "Elapsed time: depth sensor vs event camera — divergence = clock-rate mismatch"
        )
        ax5.legend(loc="upper left", fontsize=9)
        ax5.set_xlim(0, N - 1)
        ax5.grid(True, alpha=0.3)
        # Print the total drift at the end of the recording
        total_drift = float(ev_elapsed_ms[-1]) - float(depth_elapsed[-1])
        print(
            f"[LightCheck] Depth-vs-event elapsed drift over {N} frames: "
            f"{total_drift:+.2f} ms  "
            f"({'event faster' if total_drift > 0 else 'event slower'} than depth)"
        )

    fig.suptitle(
        f"Light-check temporal alignment  |  {N} frames @ {data_dir.name}  |  cam{cam_idx}",
        fontsize=12, y=1.01,
    )

    out_path = out_dir / f"light_check_align_cam{cam_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LightCheck] Saved align plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════
#  2. peaks  — per-blink delay
# ══════════════════════════════════════════════════════════════════════

def run_peaks(
    data_dir: Path,
    cam_idx: int = 0,
    smooth_k: int = 3,
    min_prominence: float = 0.15,
    fps: float = DEFAULT_FPS,
) -> None:
    """
    Detect blink peaks in both signals and plot the per-blink frame delay.

    For each peak in the RGB signal the nearest event-activity peak is found
    and the frame offset (event peak − RGB peak) is reported.  A frame offset
    of 0 means perfect alignment.

    Two output panels:
      Top    — both signals with detected peaks marked
      Bottom — per-blink frame delay (event peak − RGB peak) and its
               equivalent time delay (ms)
    """
    hdf5_dir = data_dir / "hdf5"
    out_dir = ensure_out_dir(data_dir)

    t_depth_ms, rgb_delta, t_rgb_ms = load_rgb_delta(hdf5_dir)
    event_activity, _ = load_event_activity(hdf5_dir, cam_idx)

    N = min(len(t_depth_ms), len(event_activity))
    t_depth_ms = t_depth_ms[:N]
    rgb_delta = rgb_delta[:N]
    event_activity = event_activity[:N]

    # Use RGB timestamps for wall-clock axis when available
    t_wall_ms = t_rgb_ms[:N] if t_rgb_ms is not None else t_depth_ms
    frames = np.arange(N)
    t_rel_ms = t_wall_ms - t_wall_ms[0]

    rgb_s = normalize(box_smooth(np.nan_to_num(rgb_delta), smooth_k))
    ev_s = normalize(box_smooth(event_activity, smooth_k))

    # Detect peaks with minimum prominence to ignore noise
    rgb_peaks, rgb_props = find_peaks(rgb_s, prominence=min_prominence, distance=max(3, int(fps * 0.2)))
    ev_peaks, ev_props = find_peaks(ev_s, prominence=min_prominence, distance=max(3, int(fps * 0.2)))

    # Match each RGB peak to the nearest event peak
    frame_delays = []
    ms_delays = []
    rgb_matched = []
    ev_matched = []

    for rp in rgb_peaks:
        if len(ev_peaks) == 0:
            break
        nearest_idx = int(np.argmin(np.abs(ev_peaks - rp)))
        ep = ev_peaks[nearest_idx]
        # Only count as a match if within half-second
        if abs(ep - rp) <= int(fps * 0.5):
            delay_frames = int(ep) - int(rp)
            delay_ms = t_rel_ms[ep] - t_rel_ms[rp] if ep < N and rp < N else float("nan")
            frame_delays.append(delay_frames)
            ms_delays.append(delay_ms)
            rgb_matched.append(rp)
            ev_matched.append(ep)

    n_blinks = len(frame_delays)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45)

    # ── Panel 1: signals + peaks ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(frames, rgb_s, color="#2196F3", lw=1.2, label="RGB Δ", zorder=2)
    ax1.plot(frames, ev_s, color="#FF5722", lw=1.2, alpha=0.85,
             label=f"Event activity cam{cam_idx}", zorder=2)
    ax1.scatter(rgb_peaks, rgb_s[rgb_peaks], marker="v", s=60, color="#0D47A1",
                zorder=5, label=f"RGB peaks ({len(rgb_peaks)})")
    ax1.scatter(ev_peaks, ev_s[ev_peaks], marker="^", s=60, color="#BF360C",
                zorder=5, label=f"Event peaks ({len(ev_peaks)})")
    # Draw connecting lines for matched pairs
    for rp, ep in zip(rgb_matched, ev_matched):
        ax1.annotate("", xy=(ep, ev_s[ep] + 0.06), xytext=(rp, rgb_s[rp] + 0.06),
                     arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Normalised signal")
    ax1.set_title(f"Detected blink peaks  |  {n_blinks} matched pairs")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0, N - 1)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: per-blink delay ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if n_blinks > 0:
        blink_idx = np.arange(n_blinks)
        bar_colors = ["#4CAF50" if d == 0 else "#F44336" for d in frame_delays]
        bars = ax2.bar(blink_idx, frame_delays, color=bar_colors, zorder=2, label="Frame delay")
        ax2.axhline(0, color="black", lw=1.0, ls="--")

        # Annotate each bar with ms delay
        for i, (fd, md) in enumerate(zip(frame_delays, ms_delays)):
            sign = "+" if md >= 0 else ""
            ax2.text(i, fd + (0.1 if fd >= 0 else -0.3),
                     f"{sign}{md:.1f}ms", ha="center", va="bottom" if fd >= 0 else "top",
                     fontsize=8)

        mean_delay_ms = np.nanmean(ms_delays)
        median_delay_ms = np.nanmedian(ms_delays)
        ax2.axhline(np.mean(frame_delays), color="#9C27B0", lw=1.4, ls="--",
                    label=f"mean frame delay = {np.mean(frame_delays):.2f}  "
                          f"({mean_delay_ms:+.1f} ms)")
        ax2.set_xlabel("Blink event #")
        ax2.set_ylabel("Frame delay  (event peak − RGB peak)")
        ax2.set_title(
            f"Per-blink frame delay  |  mean {np.mean(frame_delays):.2f} fr  "
            f"({mean_delay_ms:+.1f} ms)  |  median {median_delay_ms:+.1f} ms"
        )
        ax2.set_xticks(blink_idx)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        # Secondary y-axis in ms
        ms_per_frame = 1000.0 / fps
        ax2_r = ax2.twinx()
        ax2_r.set_ylim(
            ax2.get_ylim()[0] * ms_per_frame,
            ax2.get_ylim()[1] * ms_per_frame,
        )
        ax2_r.set_ylabel("Equivalent delay (ms)")
    else:
        ax2.text(0.5, 0.5, "No matching blink peaks found.\nTry lowering --min-prominence.",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12, color="gray")
        ax2.set_title("Per-blink frame delay")

    fig.suptitle(
        f"Light-check blink peak analysis  |  {N} frames @ {data_dir.name}  |  cam{cam_idx}",
        fontsize=12, y=1.01,
    )

    out_path = out_dir / f"light_check_peaks_cam{cam_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LightCheck] Saved peaks plot → {out_path}")

    # Console summary
    if n_blinks > 0:
        print(f"[LightCheck] Blink delay summary (cam{cam_idx}):")
        print(f"  matched blinks : {n_blinks}")
        print(f"  frame delays   : {frame_delays}")
        print(f"  ms delays      : {[f'{d:+.1f}' for d in ms_delays]}")
        print(f"  mean delay     : {np.mean(frame_delays):.2f} frames  ({mean_delay_ms:+.1f} ms)")
        print(f"  median delay   : {np.median(frame_delays):.2f} frames  ({median_delay_ms:+.1f} ms)")
    else:
        print(f"[LightCheck] No matching blink peaks found for cam{cam_idx}.")


# ══════════════════════════════════════════════════════════════════════
#  3. voxels  — coloured voxel strips
# ══════════════════════════════════════════════════════════════════════

def run_voxels(
    data_dir: Path,
    cam_idx: int = 0,
    min_prominence: float = 0.15,
    fps: float = DEFAULT_FPS,
) -> None:
    """
    Plot per-bin event activity at native bin timestamps against per-frame
    RGB delta, then match event-activity peaks to the nearest RGB peak that
    is immediately followed by a significant activity drop.

    The matching target — "high RGB activity right before a large drop" —
    corresponds to the moment the screen turns off (white→black), which also
    produces the largest event burst due to the OFF-bias sensitivity and the
    log-luminance asymmetry.

    Event bin timestamps are reconstructed from event_logging_start_ns
    (metadata.h5) so both signals share the same absolute millisecond axis.

    Output panels:
      1 — both signals on a shared ms time axis with matched peaks annotated
      2 — time offset: event peak time − matched RGB drop-peak time
              (using t_rgb_ms as the RGB frame timeline)
      3 — same offset but using t_global_ms (depth sensor) as the RGB
              frame timeline; comparing panels 2 and 3 shows how much the
              ~11 ms inter-sensor drift affects the apparent alignment
    """
    FALL_WINDOW = max(2, int(fps * 0.3))  # frames to look ahead for the drop

    hdf5_dir = data_dir / "hdf5"
    out_dir = ensure_out_dir(data_dir)

    # ── Clock anchor from metadata ────────────────────────────────────
    event_logging_start_ns = None
    meta_h5 = hdf5_dir / "metadata.h5"
    if meta_h5.exists():
        with h5py.File(meta_h5, "r") as mf:
            if "event_logging_start_ns" in mf.attrs:
                event_logging_start_ns = int(mf.attrs["event_logging_start_ns"])

    # ── RGB ───────────────────────────────────────────────────────────
    t_depth_ms, rgb_delta, t_rgb_ms = load_rgb_delta(hdf5_dir)
    N_rgb = len(t_depth_ms)
    t_rgb_abs = (t_rgb_ms if t_rgb_ms is not None else t_depth_ms)[:N_rgb]

    # ── Event bins at native timestamps ──────────────────────────────
    ev_h5_path = hdf5_dir / f"events_cam{cam_idx}.h5"
    if not ev_h5_path.exists():
        raise FileNotFoundError(f"events_cam{cam_idx}.h5 not found in {hdf5_dir}")
    with h5py.File(ev_h5_path, "r") as f:
        ev_frames   = f["events/frames"][:]
        t_ev_start_us = f["events/t_ev_start_us"][:]
        t_ev_end_us   = f["events/t_ev_end_us"][:]

    ev_activity   = ev_frames.astype(np.float64).mean(axis=(1, 2))
    t_ev_center_us = (t_ev_start_us.astype(np.int64) + t_ev_end_us.astype(np.int64)) // 2

    # Map event µs → absolute ms using the logging-start anchor
    if event_logging_start_ns is not None:
        t_ev_abs_ms = (event_logging_start_ns / 1e6
                       + (t_ev_center_us - t_ev_center_us[0]) / 1000.0)
    else:
        # Fallback: treat first event bin as coincident with first RGB frame
        print("[Voxels] Warning: event_logging_start_ns not found; "
              "using first RGB frame as event-stream origin.")
        t_ev_abs_ms = t_rgb_abs[0] + (t_ev_center_us - t_ev_center_us[0]) / 1000.0

    # Common origin = first RGB frame
    t0 = t_rgb_abs[0]
    t_rgb_rel   = t_rgb_abs - t0
    t_depth_rel = t_depth_ms[:N_rgb] - t0   # depth timestamps, same origin
    t_ev_rel    = t_ev_abs_ms - t0

    # ── Normalise, no smoothing ───────────────────────────────────────
    rgb_norm = normalize(np.nan_to_num(rgb_delta[:N_rgb]))
    ev_norm  = normalize(ev_activity)

    # ── Peak detection ────────────────────────────────────────────────
    dist = max(3, int(fps * 0.2))
    ev_peaks, _ = find_peaks(ev_norm, prominence=min_prominence, distance=dist)

    # RGB peaks filtered to those immediately followed by a significant drop
    rgb_all_peaks, _ = find_peaks(rgb_norm, prominence=min_prominence, distance=dist)
    rgb_peaks = []
    for p in rgb_all_peaks:
        end = min(p + FALL_WINDOW + 1, N_rgb)
        drop = (float(rgb_norm[p]) - float(np.min(rgb_norm[p + 1:end]))
                if p + 1 < end else 0.0)
        if drop >= min_prominence:
            rgb_peaks.append(p)
    rgb_peaks = np.array(rgb_peaks, dtype=int)

    # ── Match each event peak to nearest qualifying RGB drop-peak ─────
    # Using RGB (color sensor) timestamps
    ev_peak_times      = t_ev_rel[ev_peaks]
    rgb_peak_times_rgb = t_rgb_rel[rgb_peaks]   if len(rgb_peaks) else np.array([])
    rgb_peak_times_dep = t_depth_rel[rgb_peaks] if len(rgb_peaks) else np.array([])

    matches_ev_idx   = []
    matches_rgb_idx  = []
    offsets_ms_rgb   = []   # event − RGB-timestamped drop-peak
    offsets_ms_depth = []   # event − depth-timestamped drop-peak

    for ei, et in zip(ev_peaks, ev_peak_times):
        if len(rgb_peak_times_rgb) == 0:
            break
        nearest = int(np.argmin(np.abs(rgb_peak_times_rgb - et)))
        rt_rgb   = rgb_peak_times_rgb[nearest]
        rt_depth = rgb_peak_times_dep[nearest]
        if abs(et - rt_rgb) <= 500.0:  # within half a second
            matches_ev_idx.append(ei)
            matches_rgb_idx.append(rgb_peaks[nearest])
            offsets_ms_rgb.append(float(et - rt_rgb))
            offsets_ms_depth.append(float(et - rt_depth))

    n_matches = len(offsets_ms_rgb)

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    # Panel 1: signals on shared time axis with matched peaks
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_rgb_rel, rgb_norm, color="#2196F3", lw=1.0,
             label="RGB Δ  (per RGB frame)")
    ax1.plot(t_ev_rel, ev_norm, color="#FF5722", lw=1.0, alpha=0.85,
             label=f"Event activity cam{cam_idx}  (per bin, native timestamps)")
    if len(rgb_peaks):
        ax1.scatter(t_rgb_rel[rgb_peaks], rgb_norm[rgb_peaks],
                    marker="v", s=60, color="#0D47A1", zorder=5,
                    label=f"RGB drop-peaks ({len(rgb_peaks)})")
    ax1.scatter(t_ev_rel[ev_peaks], ev_norm[ev_peaks],
                marker="^", s=60, color="#BF360C", zorder=5,
                label=f"Event peaks ({len(ev_peaks)})")
    for ei, ri in zip(matches_ev_idx, matches_rgb_idx):
        ax1.annotate("",
                     xy=(t_ev_rel[ei],  ev_norm[ei]  + 0.07),
                     xytext=(t_rgb_rel[ri], rgb_norm[ri] + 0.07),
                     arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
    ax1.set_xlabel("Elapsed time (ms, relative to first RGB frame)")
    ax1.set_ylabel("Normalised signal")
    ax1.set_title(
        f"Per-bin event activity vs per-frame RGB delta  |  "
        f"{n_matches} matched pairs  |  cam{cam_idx}"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: per-match time offsets using RGB timestamps
    ax2 = fig.add_subplot(gs[1])
    if n_matches > 0:
        blink_idx  = np.arange(n_matches)
        frame_ms   = 1000.0 / fps
        bar_colors = ["#4CAF50" if abs(o) < frame_ms else "#F44336"
                      for o in offsets_ms_rgb]
        ax2.bar(blink_idx, offsets_ms_rgb, color=bar_colors, zorder=2)
        ax2.axhline(0, color="black", lw=1.0, ls="--")
        med_off_rgb  = float(np.median(offsets_ms_rgb))
        mean_off_rgb = float(np.mean(offsets_ms_rgb))
        ax2.axhline(med_off_rgb,  color="#9C27B0", lw=1.4, ls="--",
                    label=f"median = {med_off_rgb:+.1f} ms")
        ax2.axhline(mean_off_rgb, color="#FF9800", lw=1.2, ls=":",
                    label=f"mean = {mean_off_rgb:+.1f} ms")
        for i, o in enumerate(offsets_ms_rgb):
            ax2.text(i, o + (2 if o >= 0 else -2), f"{o:+.0f}",
                     ha="center", va="bottom" if o >= 0 else "top", fontsize=8)
        ax2.set_xlabel("Match #")
        ax2.set_ylabel("Event peak − RGB drop-peak  (ms)")
        ax2.set_title(
            f"Offset using t_rgb_ms (color sensor clock)  |  "
            f"median {med_off_rgb:+.1f} ms  |  mean {mean_off_rgb:+.1f} ms"
        )
        ax2.set_xticks(blink_idx)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5,
                 "No matching peaks found.\nTry lowering --min-prominence.",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=12, color="gray")
        ax2.set_title("Per-match time offset (RGB timestamps)")

    # Panel 3: same offsets but using depth timestamps for RGB frames
    ax3 = fig.add_subplot(gs[2])
    if n_matches > 0:
        blink_idx  = np.arange(n_matches)
        frame_ms   = 1000.0 / fps
        bar_colors = ["#4CAF50" if abs(o) < frame_ms else "#F44336"
                      for o in offsets_ms_depth]
        ax3.bar(blink_idx, offsets_ms_depth, color=bar_colors, zorder=2)
        ax3.axhline(0, color="black", lw=1.0, ls="--")
        med_off_dep  = float(np.median(offsets_ms_depth))
        mean_off_dep = float(np.mean(offsets_ms_depth))
        ax3.axhline(med_off_dep,  color="#9C27B0", lw=1.4, ls="--",
                    label=f"median = {med_off_dep:+.1f} ms")
        ax3.axhline(mean_off_dep, color="#FF9800", lw=1.2, ls=":",
                    label=f"mean = {mean_off_dep:+.1f} ms")
        for i, o in enumerate(offsets_ms_depth):
            ax3.text(i, o + (2 if o >= 0 else -2), f"{o:+.0f}",
                     ha="center", va="bottom" if o >= 0 else "top", fontsize=8)
        ax3.set_xlabel("Match #")
        ax3.set_ylabel("Event peak − depth-timestamped RGB drop-peak  (ms)")
        ax3.set_title(
            f"Offset using t_global_ms (depth sensor clock) for RGB frames  |  "
            f"median {med_off_dep:+.1f} ms  |  mean {mean_off_dep:+.1f} ms"
        )
        ax3.set_xticks(blink_idx)
        ax3.legend(loc="upper right", fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5,
                 "No matching peaks found.\nTry lowering --min-prominence.",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=12, color="gray")
        ax3.set_title("Per-match time offset (depth timestamps)")

    fig.suptitle(
        f"Per-bin event peak matching to RGB drop-peaks  |  "
        f"{data_dir.name}  |  cam{cam_idx}",
        fontsize=12, y=1.01,
    )

    out_path = out_dir / f"light_check_voxels_cam{cam_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LightCheck] Saved voxels plot → {out_path}")

    if n_matches > 0:
        print(f"[LightCheck] Event-to-RGB-drop-peak match summary (cam{cam_idx}):")
        print(f"  matched pairs : {n_matches}")
        print(f"  offsets using RGB timestamps  : {[f'{o:+.1f}' for o in offsets_ms_rgb]}")
        print(f"  offsets using depth timestamps: {[f'{o:+.1f}' for o in offsets_ms_depth]}")
        print(f"  mean  (RGB)   : {mean_off_rgb:+.1f} ms")
        print(f"  median (RGB)  : {med_off_rgb:+.1f} ms")
        print(f"  mean  (depth) : {mean_off_dep:+.1f} ms")
        print(f"  median (depth): {med_off_dep:+.1f} ms")
    else:
        print(f"[LightCheck] No matching peaks found for cam{cam_idx}.")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "subcommand",
        choices=["align", "peaks", "voxels", "all"],
        help="Which analysis to run",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/light_check"),
        help="Path to the light_check recording directory (default: data/light_check)",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        metavar="IDX",
        help="Event camera index to analyse (default: 0)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        metavar="K",
        help="Box-filter window size for smoothing (default: 3; use 1 to disable)",
    )
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=0.15,
        help="Minimum peak prominence for blink detection (default: 0.15)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve relative paths from the script's parent (3d_reconstruction/)
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent.parent / data_dir
    data_dir = data_dir.resolve()

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.subcommand in ("align", "all"):
        run_align(data_dir, cam_idx=args.cam, smooth_k=args.smooth)

    if args.subcommand in ("peaks", "all"):
        run_peaks(
            data_dir,
            cam_idx=args.cam,
            smooth_k=args.smooth,
            min_prominence=args.min_prominence,
        )

    if args.subcommand in ("voxels", "all"):
        run_voxels(data_dir, cam_idx=args.cam, min_prominence=args.min_prominence)


if __name__ == "__main__":
    main()
