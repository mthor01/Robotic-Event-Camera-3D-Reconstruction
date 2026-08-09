#!/usr/bin/env python3
"""
Plot consecutive pose timestamp differences.

For a recording's raw_poses.h5 this script reads the published (t_ns) and
received (t_recv_ns) timestamps and plots, for every consecutive pair i / i+1:

    Δt_pub   [ms]  = t_ns[i+1]      − t_ns[i]       (robot-side publish interval)
    Δt_recv  [ms]  = t_recv_ns[i+1] − t_recv_ns[i]  (recorder-side receive interval)

Both series are shown in a single figure so you can spot dropped messages
(large spikes), bursts, or systematic drift at a glance.

Output: viz_and_tests/plots/pose_intervals_<object_name>.png

Usage:
    python pose_interval_plot.py --data_dir data/real/banana
    python pose_interval_plot.py --data_dir data/real/banana --out_dir /tmp/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from 3d_reconstruction when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_timestamps(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load t_ns and t_recv_ns from <data_dir>/hdf5/raw_poses.h5."""
    h5_path = data_dir / "hdf5" / "raw_poses.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"raw_poses.h5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        t_ns = f["t_ns"][:]
        t_recv_ns = f["t_recv_ns"][:]
    return t_ns.astype(np.int64), t_recv_ns.astype(np.int64)


def consecutive_diff_ms(timestamps_ns: np.ndarray) -> np.ndarray:
    """Return the difference between every consecutive pair in milliseconds."""
    return np.diff(timestamps_ns) / 1e6


def plot_intervals(
    data_dir: Path,
    out_dir: Path,
) -> Path:
    """
    Build the interval plot and save it to out_dir.

    Returns the path to the saved PNG.
    """
    t_ns, t_recv_ns = load_timestamps(data_dir)
    object_name = data_dir.name

    dt_pub_ms   = consecutive_diff_ms(t_ns)
    dt_recv_ms  = consecutive_diff_ms(t_recv_ns)
    indices     = np.arange(1, len(t_ns))   # pair (i-1, i) labelled by the later index

    # ── statistics ────────────────────────────────────────────────────
    def stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(),
            "median": np.median(arr),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "n_large": int((arr > 50).sum()),  # gaps > 50 ms
        }

    s_pub  = stats(dt_pub_ms)
    s_recv = stats(dt_recv_ms)

    # ── figure layout ─────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    fig.suptitle(
        f"Consecutive pose timestamp intervals — {object_name}\n"
        f"({len(t_ns)} poses total)",
        fontsize=13,
        fontweight="bold",
    )

    colours = {"pub": "#1f77b4", "recv": "#ff7f0e"}

    for ax, dt_ms, s, label, colour in [
        (axes[0], dt_pub_ms,  s_pub,  "Published (t_ns) — robot-side",          colours["pub"]),
        (axes[1], dt_recv_ms, s_recv, "Received  (t_recv_ns) — recorder-side",  colours["recv"]),
    ]:
        ax.plot(indices, dt_ms, linewidth=0.7, color=colour, alpha=0.85, label=label)

        # Median reference line
        ax.axhline(s["median"], color="black", linewidth=0.9, linestyle="--",
                   label=f"median = {s['median']:.2f} ms")

        # Highlight gaps > 50 ms
        large_mask = dt_ms > 50
        if large_mask.any():
            ax.scatter(indices[large_mask], dt_ms[large_mask],
                       color="red", s=18, zorder=5, label=f"gaps > 50 ms ({s['n_large']})")

        ax.set_ylabel("Δt [ms]", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linewidth=0.4, alpha=0.5)

        # Annotation box with summary stats
        stat_text = (
            f"mean={s['mean']:.2f}  median={s['median']:.2f}  "
            f"std={s['std']:.2f}  min={s['min']:.2f}  max={s['max']:.2f} ms"
        )
        ax.text(
            0.01, 0.97, stat_text,
            transform=ax.transAxes,
            fontsize=7.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    axes[-1].set_xlabel("Pose index", fontsize=10)

    # ── save ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pose_intervals_{object_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[pose_interval_plot] Saved → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot consecutive pose timestamp intervals for a recording."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the recording directory (must contain hdf5/raw_poses.h5).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Directory to save the PNG (default: viz_and_tests/plots/ "
            "relative to this script)."
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir  = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path(__file__).resolve().parent / "plots"
    )

    plot_intervals(data_dir, out_dir)


if __name__ == "__main__":
    main()
