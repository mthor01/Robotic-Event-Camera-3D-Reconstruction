#!/usr/bin/env python3
"""Plot EE speed vs event activity over time, including a middle 3 s window.

This script uses the original raw pose samples from ``raw_poses.h5`` so the EE
speed is computed from the true pose timestamps instead of frame-discretized
pose assignments.

The output includes:
  - full-sequence plot of EE speed vs event activity
  - zoomed plot over the middle 3 s of the recording
  - optional time-shift applied to the raw pose timestamps (default: 0 ms)

Usage examples:
    python3 ee_speed_event_activity.py --data_dir data/real/temporal_check_01
    python3 ee_speed_event_activity.py --data_dir data/real/temporal_check_01 --pose_offset_ms 12.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_raw_pose_times_and_transforms(raw_pose_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not raw_pose_path.exists():
        raise FileNotFoundError(f"Missing raw pose file: {raw_pose_path}")

    with h5py.File(raw_pose_path, "r") as f:
        if "t_recv_ns" in f:
            t_ns = f["t_recv_ns"][:].astype(np.int64)
        elif "t_ns" in f:
            t_ns = f["t_ns"][:].astype(np.int64)
        else:
            raise KeyError(f"No pose timestamps found in {raw_pose_path}")

        if "ee_T" in f:
            ee_T = f["ee_T"][:].astype(np.float64)
        else:
            raise KeyError(f"No ee_T found in {raw_pose_path}")

    if len(t_ns) != len(ee_T):
        raise ValueError(
            f"Pose timestamps and ee_T length mismatch: {len(t_ns)} vs {len(ee_T)}"
        )

    return t_ns, ee_T


def compute_pose_speed(t_ns: np.ndarray, ee_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ns, speed_m_per_s) from raw pose samples."""
    if len(t_ns) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    positions = ee_T[:, :3, 3]
    dt_s = np.diff(t_ns.astype(np.float64)) / 1e9
    dt_s[dt_s <= 0] = np.nan

    delta_pos = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = delta_pos / dt_s

    time_mid_ns = ((t_ns[:-1] + t_ns[1:]) / 2.0).astype(np.int64)
    return time_mid_ns, speed


def load_event_activity(event_h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not event_h5_path.exists():
        raise FileNotFoundError(f"Missing event HDF5 file: {event_h5_path}")

    with h5py.File(event_h5_path, "r") as f:
        events_group = f["events"]
        if "frames" not in events_group:
            raise KeyError(f"No 'events/frames' in {event_h5_path}")

        frames = events_group["frames"][:]
        if "t_ev_start_us" not in events_group or "t_ev_end_us" not in events_group:
            raise KeyError(f"Missing event time arrays in {event_h5_path}")

        t_start_us = events_group["t_ev_start_us"][:].astype(np.int64)
        t_end_us = events_group["t_ev_end_us"][:].astype(np.int64)

    # Event-activity proxy: per-frame pixel std, same as the temporal alignment tests.
    activity = frames.astype(np.float64).std(axis=(1, 2))
    frame_mid_ns = ((t_start_us + t_end_us) // 2) * 1000
    return frame_mid_ns.astype(np.int64), activity.astype(np.float64)


def apply_pose_offset_ns(t_ns: np.ndarray, offset_ms: float) -> np.ndarray:
    return t_ns + int(round(offset_ms * 1e6))


def _cut_window(t_ns: np.ndarray, center_ns: int, half_span_ns: int) -> tuple[int, int]:
    start = center_ns - half_span_ns
    end = center_ns + half_span_ns
    return start, end


def _select_between(t_ns: np.ndarray, t0: int, t1: int) -> np.ndarray:
    return np.logical_and(t_ns >= t0, t_ns <= t1)


def build_middle_3s_window_ns(t_pose_ns: np.ndarray, t_event_ns: np.ndarray) -> tuple[int, int]:
    """Return a 3 s window centered on the common overlap of the two time series.

    If the common time span is shorter than 3 s, the window collapses to the
    available span instead of producing an empty selection.
    """
    pose_min = int(t_pose_ns.min())
    pose_max = int(t_pose_ns.max())
    event_min = int(t_event_ns.min())
    event_max = int(t_event_ns.max())

    t0 = max(pose_min, event_min)
    t1 = min(pose_max, event_max)
    if t1 <= t0:
        t0 = min(pose_min, event_min)
        t1 = max(pose_max, event_max)

    span_ns = max(1, t1 - t0)
    center_ns = (t0 + t1) // 2
    half_window_ns = min(int(1.5e9), span_ns // 2)
    if half_window_ns <= 0:
        half_window_ns = span_ns // 2 if span_ns > 0 else 1

    start = center_ns - half_window_ns
    end = center_ns + half_window_ns
    return start, end


def plot_sequence(seq_dir: Path, pose_offset_ms: float = 0.0, output_name: str = "ee_speed_event_activity.png") -> Path:
    hdf5_dir = seq_dir / "hdf5"
    raw_pose_path = hdf5_dir / "raw_poses.h5"
    event_h5_path = hdf5_dir / "events_cam0.h5"

    t_pose_ns, ee_T = load_raw_pose_times_and_transforms(raw_pose_path)
    t_pose_ns = apply_pose_offset_ns(t_pose_ns, pose_offset_ms)
    t_pose_speed_ns, speed = compute_pose_speed(t_pose_ns, ee_T)

    t_event_ns, event_activity = load_event_activity(event_h5_path)

    if len(t_pose_speed_ns) == 0 or len(t_event_ns) == 0:
        raise ValueError(f"Not enough pose or event samples in {seq_dir}")

    full_t_pose = (t_pose_speed_ns - t_pose_speed_ns[0]) / 1e9
    full_t_event = (t_event_ns - t_event_ns[0]) / 1e9

    out_dir = _ensure_dir(seq_dir / "pose_plots")
    out_path = out_dir / output_name

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False, constrained_layout=True)

    def plot_panel(ax, title: str, t_pose_plot: np.ndarray, speed_plot: np.ndarray,
                   t_event_plot: np.ndarray, event_plot: np.ndarray,
                   x_label: str = "Time (s)") -> None:
        if len(t_pose_plot) == 0 or len(t_event_plot) == 0:
            ax.text(0.5, 0.5, "No data in this time window", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return

        ax2 = ax.twinx()

        ax.plot((t_pose_plot - t_pose_plot[0]) / 1e9, speed_plot, color="#2d6cdf", lw=2.0, label="EE speed")
        ax.set_ylabel("EE speed (m/s)", color="#2d6cdf")
        ax.tick_params(axis="y", colors="#2d6cdf")
        ax.set_title(title)

        ax2.plot((t_event_plot - t_event_plot[0]) / 1e9, event_plot, color="#d64b4b", lw=1.8, alpha=0.95, label="Event activity")
        ax2.set_ylabel("Event activity (std)", color="#d64b4b")
        ax2.tick_params(axis="y", colors="#d64b4b")

        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=9)

    plot_panel(axes[0], "Whole sequence", t_pose_speed_ns, speed, t_event_ns, event_activity)

    w0, w1 = build_middle_3s_window_ns(t_pose_speed_ns, t_event_ns)
    pose_mask = _select_between(t_pose_speed_ns, w0, w1)
    event_mask = _select_between(t_event_ns, w0, w1)

    if not pose_mask.any() or not event_mask.any():
        pose_plot = t_pose_speed_ns
        event_plot = t_event_ns
        speed_plot = speed
        event_activity_plot = event_activity
        win_title = "Middle 3 s window not available; using full sequence"
    else:
        pose_plot = t_pose_speed_ns[pose_mask]
        event_plot = t_event_ns[event_mask]
        speed_plot = speed[pose_mask]
        event_activity_plot = event_activity[event_mask]
        win_title = f"Middle 3 s window ({(w1 - w0)/1e9:.1f} s)"

    plot_panel(axes[1], win_title, pose_plot, speed_plot, event_plot, event_activity_plot)

    fig.suptitle(f"EE speed and event activity — {seq_dir.name}\npose offset = {pose_offset_ms:+.2f} ms", fontsize=12, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot raw-pose EE speed vs event activity for the whole sequence and the middle 3 s window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Sequence directory containing hdf5/raw_poses.h5 and hdf5/events_cam0.h5")
    parser.add_argument("--pose_offset_ms", type=float, default=0.0, help="Optional timestamp offset applied to raw poses before plotting (default: 0 ms).")
    parser.add_argument("--output_name", type=str, default="ee_speed_event_activity.png", help="Output filename in the sequence's pose_plots directory.")
    args = parser.parse_args()

    seq_dir = Path(args.data_dir).resolve()
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    out_path = plot_sequence(seq_dir, pose_offset_ms=args.pose_offset_ms, output_name=args.output_name)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
