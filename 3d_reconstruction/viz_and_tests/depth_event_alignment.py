#!/usr/bin/env python3
"""Legacy temporal-alignment diagnostic: RGB rate-of-change vs event activity.

Both signals dip at every direction reversal during a TemporalAlignmentAgent
recording:
  - RGB ROC: mean |gray[i] - gray[i-1]| over all pixels.
    Falls to near-zero when the robot pauses at a turning point.
  - Event activity: per-frame pixel std of the event frame.
    Also dips at reversals because the camera is momentarily still.

Matching trough positions estimates a frame-level stream offset without pose
data. Current recordings use hardware-trigger timestamps and do not require
this estimate; this utility remains useful for inspecting older recordings.

Usage:
    python depth_event_alignment.py --data_dir data/temporal_check
    python depth_event_alignment.py --data_root data/real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconstruction_config import DATA_ROOT as _DATA_ROOT


# ──────────────────────────────────────────────────────────────────────
#  Helpers (shared with pose_time_analysis)
# ──────────────────────────────────────────────────────────────────────

def box_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.astype(np.float64).copy()
    return np.convolve(arr.astype(np.float64), np.ones(k) / k, mode="same")


def find_troughs(sig: np.ndarray, min_prominence_frac: float = 0.15) -> np.ndarray:
    if len(sig) < 3:
        return np.array([], dtype=np.float64)
    rng = float(sig.max() - sig.min())
    if rng == 0:
        return np.array([], dtype=np.float64)
    peaks, _ = find_peaks(-sig, prominence=min_prominence_frac * rng)
    return peaks.astype(np.float64)


def unit_norm(a: np.ndarray) -> np.ndarray:
    mx = np.abs(a).max()
    return a / mx if mx > 0 else a


# ──────────────────────────────────────────────────────────────────────
#  Signal extraction
# ──────────────────────────────────────────────────────────────────────

def compute_rgb_roc(rgb_ds, batch: int = 32) -> np.ndarray:
    """
    Mean absolute frame-to-frame change of the greyscale-converted RGB image.

    Returns array of length N where entry 0 is 0 and entry i is the
    mean |gray[i] - gray[i-1]|.
    """
    N = rgb_ds.shape[0]
    roc = np.zeros(N, dtype=np.float64)  # index 0 left as 0

    def to_gray(frame):
        f = frame.astype(np.float32)
        # standard luminance weights for BGR or RGB (order doesn't affect ROC)
        return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]

    prev = to_gray(rgb_ds[0])
    for start in range(1, N, batch):
        end = min(start + batch, N)
        chunk = rgb_ds[start:end]
        for j in range(end - start):
            gray = to_gray(chunk[j])
            roc[start + j] = np.mean(np.abs(gray - prev))
            prev = gray
    return roc


def compute_event_activity(ev_frames: np.ndarray) -> np.ndarray:
    """Per-frame pixel std of event frames."""
    return ev_frames.astype(np.float64).std(axis=(1, 2))


# ──────────────────────────────────────────────────────────────────────
#  Main analysis
# ──────────────────────────────────────────────────────────────────────

def run_analysis(seq_dir: Path, smooth_k: int = 3, cam_idx: int = 0) -> None:
    hdf5_dir = seq_dir / "hdf5"

    rs_path = hdf5_dir / "realsense.h5"
    ev_path = hdf5_dir / f"events_cam{cam_idx}.h5"
    for p in (rs_path, ev_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    out_dir = seq_dir / "pose_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load signals ──────────────────────────────────────────────────
    print(f"[{seq_dir.name}] Loading RGB frames …")
    with h5py.File(rs_path, "r") as f:
        rgb_roc = compute_rgb_roc(f["rgb"])
        N_rgb = f["rgb"].shape[0]
        t_hw_ms = f["t_hw_ms"][:] if "t_hw_ms" in f else None

    print(f"[{seq_dir.name}] Loading event frames …")
    with h5py.File(ev_path, "r") as f:
        ev_frames = f["events/frames"][:]
        t_ev_end_us = f["events/t_ev_end_us"][:]

    N_ev = len(ev_frames)
    L = min(N_rgb, N_ev)

    event_activity = compute_event_activity(ev_frames[:L])
    rgb_roc = rgb_roc[:L]

    # Trim the leading/trailing 1 % of frames (camera spin-up, end noise)
    TRIM = max(1, int(0.01 * L))
    s, e = TRIM, max(TRIM + 1, L - TRIM)
    rgb_roc = rgb_roc[s:e]
    event_activity = event_activity[s:e]
    L = len(rgb_roc)
    frames = np.arange(L)

    # ── Smooth + centre ───────────────────────────────────────────────
    dr_c = rgb_roc - rgb_roc.mean()
    ev_c = event_activity - event_activity.mean()
    dr_s = box_smooth(dr_c, smooth_k)
    ev_s = box_smooth(ev_c, smooth_k)

    # ── Trough detection ─────────────────────────────────────────────
    troughs_dr = find_troughs(dr_s)
    troughs_ev = find_troughs(ev_s)

    # ── Greedy nearest-neighbour matching ────────────────────────────
    offsets, matched_dr, matched_ev_list = [], [], []
    if len(troughs_dr) > 0 and len(troughs_ev) > 0:
        candidates = sorted(
            (abs(float(a) - float(b)), ia, ib)
            for ia, a in enumerate(troughs_dr)
            for ib, b in enumerate(troughs_ev)
        )
        used_dr, used_ev = set(), set()
        for _d, ia, ib in candidates:
            if ia not in used_dr and ib not in used_ev:
                used_dr.add(ia); used_ev.add(ib)
                offsets.append(float(troughs_ev[ib] - troughs_dr[ia]))
                matched_dr.append(float(troughs_dr[ia]))
                matched_ev_list.append(float(troughs_ev[ib]))

    offsets = np.array(offsets) if offsets else np.array([])
    int_offsets = np.round(offsets).astype(int) if len(offsets) > 0 else np.array([], dtype=int)

    # Frame period (use hardware timestamps if available)
    if t_hw_ms is not None and len(t_hw_ms) > 1:
        frame_dt_ms = float(np.median(np.diff(t_hw_ms)))
    else:
        frame_dt_ms = 1000.0 / 30.0
    offsets_ms = offsets * frame_dt_ms

    # Filter obviously wrong matches (> 500 ms)
    if len(offsets_ms) > 0:
        mask = np.abs(offsets_ms) < 500.0
        offsets = offsets[mask]; int_offsets = int_offsets[mask]
        offsets_ms = offsets_ms[mask]
        matched_dr = [v for v, m in zip(matched_dr, mask) if m]
        matched_ev_list = [v for v, m in zip(matched_ev_list, mask) if m]

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DEPTH ↔ EVENT TEMPORAL ALIGNMENT")
    print("=" * 60)
    print(f"  Recording : {seq_dir}")
    print(f"  RGB frames: {N_rgb}   Event frames: {N_ev}   Common: {L}")
    print(f"  RGB ROC troughs   : {len(troughs_dr)}")
    print(f"  Event activity troughs: {len(troughs_ev)}")
    if len(int_offsets) > 0:
        med = float(np.median(int_offsets))
        print(f"  Matched pairs : {len(int_offsets)}")
        print(f"  Median offset : {med:+.1f} fr  ({np.median(offsets_ms):+.1f} ms)")
        print(f"  Std           : {offsets_ms.std():.1f} ms")
        label = "✓ GOOD" if abs(med) < 1 else ("~ OK (±1)" if abs(med) <= 1 else f"✗ OFF ({med:+.0f} fr)")
        print(f"  Verdict: {label}")
    else:
        print("  No matched pairs found.")
    print("=" * 60)

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1.2],
                           hspace=0.38, wspace=0.30)
    col_dr = "#2066a8"
    col_ev = "#d6604d"

    # Panel 1: raw signals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(frames, rgb_roc, color=col_dr, lw=0.9, label="RGB ROC (raw)")
    ax1.set_ylabel("Mean |Δgray| (uint8)", color=col_dr)
    ax1r = ax1.twinx()
    ax1r.plot(frames, event_activity, color=col_ev, lw=0.7, alpha=0.8,
              label="Event activity (raw)")
    ax1r.set_ylabel("Event std", color=col_ev)
    ax1.set_title(f"RGB rate-of-change & event activity — {seq_dir.name}", fontsize=11)
    ax1.set_xlabel("Frame index")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8)

    # Panel 2: normalised overlay with detected troughs
    ax2 = fig.add_subplot(gs[1, :])
    drn = unit_norm(dr_s)
    evn = unit_norm(ev_s)
    ax2.plot(frames, drn, color=col_dr, lw=1.2, label="RGB ROC (norm, smoothed)")
    ax2.plot(frames, evn, color=col_ev, lw=1.2, alpha=0.8,
             label="Event activity (norm, smoothed)")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    for td in troughs_dr:
        i = int(min(round(td), L - 1))
        ax2.plot(td, drn[i], "v", color=col_dr, ms=6, zorder=5)
    for te in troughs_ev:
        i = int(min(round(te), L - 1))
        ax2.plot(te, evn[i], "v", color=col_ev, ms=6, zorder=5)
    ax2.set_ylabel("Normalised")
    ax2.set_xlabel("Frame index")
    ax2.set_title(
        f"RGB ROC vs event activity (▼ = troughs)  "
        f"|  RGB troughs: {len(troughs_dr)}  event troughs: {len(troughs_ev)}",
        fontsize=10,
    )
    ax2.legend(fontsize=8)

    # Panel 3: offset histogram
    ax3 = fig.add_subplot(gs[2, 0])
    if len(int_offsets) > 0:
        lo, hi = int_offsets.min(), int_offsets.max()
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        ax3.hist(int_offsets, bins=bins, color="#8da0cb", edgecolor="k",
                 linewidth=0.8, rwidth=0.85)
        ax3.axvline(0, color="red", lw=1.2, ls="--", label="zero")
        ax3.axvline(float(np.median(int_offsets)), color="orange", lw=1.2,
                    label=f"med={np.median(int_offsets):+.0f}fr")
        ax3.set_xlabel("Frame offset (RGB − event)")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No matched pairs", ha="center", va="center",
                 transform=ax3.transAxes, color="red")
    ax3.set_title("Offset distribution", fontsize=10)

    # Panel 4: offset over time
    ax4 = fig.add_subplot(gs[2, 1])
    if len(int_offsets) > 0:
        dot_c = ["#4daf4a" if o == 0 else ("#2066a8" if abs(o) <= 1 else "#ff7f00")
                 for o in int_offsets]
        ax4.scatter(matched_dr, int_offsets, color=dot_c, edgecolors="k",
                    linewidths=0.5, s=40, zorder=3)
        ax4.axhline(0, color="gray", lw=0.5, ls=":")
        ax4.axhline(float(np.median(int_offsets)), color="red", lw=1.0, ls="--")
        ax4.set_xlabel("Frame of RGB ROC trough")
        ax4.set_ylabel("Offset (frames)")
        med = float(np.median(int_offsets))
        label = "✓ GOOD" if abs(med) < 1 else ("~ OK" if abs(med) <= 1 else "✗ OFF")
        ax4.set_title(
            f"Offset over time  |  n={len(int_offsets)}  "
            f"med={np.median(offsets_ms):+.1f}ms  {label}",
            fontsize=9,
        )
    else:
        ax4.set_title("Offset over time", fontsize=10)

    fig.suptitle(
        "RGB ROC ↔ Event Activity Temporal Alignment",
        fontsize=13, fontweight="bold", y=0.98,
    )
    out_path = out_dir / "depth_event_alignment.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

def find_recordings(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "hdf5" / "realsense.h5").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth ROC ↔ event activity temporal alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", nargs="+", type=str, default=None,
                        help="Specific recording directory(ies)")
    parser.add_argument("--data_root", type=str, default=str(_DATA_ROOT),
                        help="Root containing recording subdirs")
    parser.add_argument("--cam", type=int, default=0,
                        help="Event camera index")
    parser.add_argument("--smooth_k", type=int, default=3,
                        help="Box-smoothing kernel width (frames)")
    args = parser.parse_args()

    if args.data_dir:
        dirs = [Path(d) for d in args.data_dir]
    else:
        dirs = find_recordings(Path(args.data_root))
        if not dirs:
            print(f"No recordings found under {args.data_root}")
            return

    for d in dirs:
        try:
            run_analysis(d, smooth_k=args.smooth_k, cam_idx=args.cam)
        except FileNotFoundError as exc:
            print(f"[skip] {d.name}: {exc}")
        except Exception as exc:
            print(f"[error] {d.name}: {exc}")
            raise


if __name__ == "__main__":
    main()
