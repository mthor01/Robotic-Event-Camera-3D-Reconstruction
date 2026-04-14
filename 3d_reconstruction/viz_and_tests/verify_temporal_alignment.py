"""
Temporal alignment verification for robot recording data.

Proves or disproves that the interpolated robot poses stored in poses.h5 are
well-aligned in time with the depth/RGB frames stored in realsense.h5.

Key metric: nearest_offset_ms (saved during recording) = time distance from
each frame timestamp to the closest raw robot-pose sample.  If this is
consistently below half a frame interval, every frame has an actual pose
observation nearby and the interpolation is trustworthy.

Usage:
    python verify_temporal_alignment.py data/real/1
    python verify_temporal_alignment.py data/real/1 --output my_check.png
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────


def load_data(hdf5_dir: Path):
    rs_path = hdf5_dir / "realsense.h5"
    poses_path = hdf5_dir / "poses.h5"
    meta_path = hdf5_dir / "metadata.h5"

    for p in (rs_path, poses_path):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    with h5py.File(rs_path, "r") as f:
        t_sys_ns = f["t_sys_ns"][:]        # (N,) int64  system clock, ns
        t_hw_ms  = f["t_hw_ms"][:]         # (N,) float64 hardware clock, ms
        frame_no = f["frame_number"][:]    # (N,) int64

    with h5py.File(poses_path, "r") as f:
        ee_T      = f["ee_T"][:]           # (N, 4, 4)
        joints    = f["joint_positions"][:] # (N, 7)
        offset_ms = f["nearest_offset_ms"][:] # (N,)

    fps = 30.0
    if meta_path.exists():
        with h5py.File(meta_path, "r") as f:
            fps = float(f.attrs.get("fps", 30.0))
            raw_poses = int(f.attrs.get("raw_poses_received", -1))
    else:
        raw_poses = -1

    return t_sys_ns, t_hw_ms, frame_no, ee_T, joints, offset_ms, fps, raw_poses


def compute_metrics(t_sys_ns, t_hw_ms, frame_no, ee_T, joints, offset_ms, fps):
    N = len(t_sys_ns)

    # Time axis starting from 0 (seconds)
    t_s = (t_sys_ns - t_sys_ns[0]) / 1e9

    # Frame intervals from hardware timestamps (more accurate than sys clock)
    dt_hw_ms = np.diff(t_hw_ms)          # (N-1,)
    dt_sys_ms = np.diff(t_sys_ns) / 1e6  # (N-1,)

    # Dropped frames (gaps in hardware frame counter)
    frame_gaps = np.diff(frame_no.astype(np.int64))   # should be all 1
    dropped = int(np.sum(frame_gaps > 1))
    total_missing = int(np.sum(frame_gaps - 1))

    # EE position (last column of 4×4 transform)
    ee_pos = ee_T[:, :3, 3]              # (N, 3) in metres

    # EE speed: distance between consecutive positions, divided by hw time step
    dt_for_vel = dt_hw_ms / 1000.0       # (N-1,) in seconds
    ee_speed = (
        np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)
        / np.where(dt_for_vel > 0, dt_for_vel, np.nan)
    )                                    # m/s

    # Expected frame interval
    target_dt_ms = 1000.0 / fps

    # Summary statistics
    duration_s = t_s[-1] - t_s[0]
    actual_fps = (N - 1) / duration_s if duration_s > 0 else 0.0
    half_frame_ms = 1000.0 / (2.0 * actual_fps) if actual_fps > 0 else 16.7

    off_median  = float(np.median(offset_ms))
    off_p95     = float(np.percentile(offset_ms, 95))
    off_max     = float(np.max(offset_ms))

    dt_median = float(np.median(dt_hw_ms))
    dt_std    = float(np.std(dt_hw_ms))

    # Verdict: accept if median < ½ frame and p95 < 1 frame
    aligned = off_median < half_frame_ms and off_p95 < target_dt_ms
    verdict = "WELL-ALIGNED" if aligned else "POTENTIALLY MISALIGNED"

    return dict(
        N=N, t_s=t_s, dt_hw_ms=dt_hw_ms, dt_sys_ms=dt_sys_ms,
        frame_gaps=frame_gaps, dropped=dropped, total_missing=total_missing,
        ee_pos=ee_pos, ee_speed=ee_speed,
        target_dt_ms=target_dt_ms, duration_s=duration_s, actual_fps=actual_fps,
        half_frame_ms=half_frame_ms,
        off_median=off_median, off_p95=off_p95, off_max=off_max,
        dt_median=dt_median, dt_std=dt_std,
        aligned=aligned, verdict=verdict,
    )


def make_figure(t_s, offset_ms, dt_hw_ms, ee_pos, ee_speed, joints, m, fps,
                raw_poses, object_name, out_path):
    """Build and save the 3×3 diagnostic figure."""

    verdict_color = "#1a7431" if m["aligned"] else "#b22222"
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#f5f5f5")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    suptitle = (
        f"Temporal Alignment Verification — recording: {object_name}\n"
        f"Verdict: {m['verdict']}  |  {m['N']} frames, "
        f"{m['duration_s']:.1f} s @ {m['actual_fps']:.1f} fps  |  "
        f"raw poses received: {raw_poses}"
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold",
                 color=verdict_color, y=0.995)

    # ── 1  nearest_offset_ms over time ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.fill_between(t_s, offset_ms, alpha=0.25, color="steelblue")
    ax1.plot(t_s, offset_ms, color="steelblue", lw=0.7, alpha=0.85)
    ax1.axhline(m["off_median"], color="navy", ls="--", lw=1.5,
                label=f"median  {m['off_median']:.1f} ms")
    ax1.axhline(m["off_p95"], color="darkorange", ls="-.", lw=1.5,
                label=f"p95     {m['off_p95']:.1f} ms")
    ax1.axhline(m["half_frame_ms"], color="red", ls=":", lw=1.8,
                label=f"½ frame {m['half_frame_ms']:.1f} ms  (accept threshold)")
    ax1.set_xlabel("Recording time (s)")
    ax1.set_ylabel("Pose offset (ms)")
    ax1.set_title("Nearest raw-pose offset per frame  [PRIMARY METRIC]",
                  fontweight="bold")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    _color_background(ax1, m["off_median"] < m["half_frame_ms"])

    # ── 2  offset_ms histogram ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(offset_ms, bins=60, color="steelblue", edgecolor="white", alpha=0.80)
    ax2.axvline(m["off_median"], color="navy", ls="--", lw=1.5,
                label=f"median {m['off_median']:.1f} ms")
    ax2.axvline(m["off_p95"], color="darkorange", ls="-.", lw=1.5,
                label=f"p95    {m['off_p95']:.1f} ms")
    ax2.axvline(m["half_frame_ms"], color="red", ls=":", lw=1.8,
                label=f"½ frame {m['half_frame_ms']:.1f} ms")
    ax2.axvline(m["target_dt_ms"], color="green", ls="--", lw=1.2,
                label=f"1 frame {m['target_dt_ms']:.1f} ms")
    ax2.set_xlabel("Offset (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("Offset distribution")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    stats_txt = (
        f"min:    {offset_ms.min():.2f} ms\n"
        f"median: {m['off_median']:.2f} ms\n"
        f"p95:    {m['off_p95']:.2f} ms\n"
        f"max:    {m['off_max']:.2f} ms\n"
        f"½ frame:{m['half_frame_ms']:.2f} ms\n"
        f"1 frame:{m['target_dt_ms']:.2f} ms"
    )
    ax2.text(0.97, 0.97, stats_txt, transform=ax2.transAxes,
             fontsize=7.5, va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    # ── 3  hardware frame interval over time ──────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(t_s[1:], dt_hw_ms, color="teal", lw=0.7, alpha=0.80)
    ax3.axhline(m["dt_median"], color="darkgreen", ls="--", lw=1.5,
                label=f"median {m['dt_median']:.1f} ms")
    ax3.axhline(m["target_dt_ms"], color="gray", ls=":", lw=1.5,
                label=f"target {m['target_dt_ms']:.1f} ms ({fps:.0f} fps)")
    ax3.set_xlabel("Recording time (s)")
    ax3.set_ylabel("Frame interval (ms)")
    ax3.set_title("Hardware frame-to-frame interval  [camera jitter check]")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4  frame interval histogram ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(dt_hw_ms, bins=60, color="teal", edgecolor="white", alpha=0.80)
    ax4.axvline(m["target_dt_ms"], color="red", ls="--", lw=1.5,
                label=f"target {m['target_dt_ms']:.1f} ms")
    ax4.set_xlabel("Frame interval (ms)")
    ax4.set_ylabel("Count")
    ax4.set_title("Frame interval distribution")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    jitter_txt = (
        f"median: {m['dt_median']:.2f} ms\n"
        f"std:    {m['dt_std']:.2f} ms\n"
        f"min:    {dt_hw_ms.min():.2f} ms\n"
        f"max:    {dt_hw_ms.max():.2f} ms"
    )
    ax4.text(0.97, 0.97, jitter_txt, transform=ax4.transAxes,
             fontsize=7.5, va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    # ── 5  EE position XYZ over time ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    colors_xyz = ["#e74c3c", "#27ae60", "#2980b9"]
    for i, (label, col) in enumerate(zip(["x", "y", "z"], colors_xyz)):
        ax5.plot(t_s, ee_pos[:, i], color=col, lw=0.9, alpha=0.85, label=label)
    ax5.set_xlabel("Recording time (s)")
    ax5.set_ylabel("EE position (m)")
    ax5.set_title("End-effector XYZ trajectory  [smooth = good pose alignment]")
    ax5.legend(fontsize=9, loc="upper right")
    ax5.grid(True, alpha=0.3)

    # ── 6  EE speed over time ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(t_s[1:], ee_speed * 100, color="purple", lw=0.8, alpha=0.80)
    ax6.set_xlabel("Recording time (s)")
    ax6.set_ylabel("EE speed (cm/s)")
    ax6.set_title("Inferred EE speed\n(discontinuities → misalignment)")
    ax6.grid(True, alpha=0.3)

    # ── Save ──────────────────────────────────────────────────────────
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _color_background(ax, good: bool):
    """Tint an axes background green/red to emphasise pass/fail."""
    color = "#e8f5e9" if good else "#ffebee"
    ax.set_facecolor(color)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify temporal alignment between robot poses and camera frames."
    )
    parser.add_argument(
        "object_dir",
        help="Path to the recording directory, e.g. data/real/1",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output PNG path (default: <object_dir>/temporal_alignment.png)",
    )
    args = parser.parse_args()

    object_dir = Path(args.object_dir)
    hdf5_dir = object_dir / "hdf5"
    out_path = Path(args.output) if args.output else object_dir / "temporal_alignment.png"

    print(f"Loading data from {hdf5_dir} ...")
    t_sys_ns, t_hw_ms, frame_no, ee_T, joints, offset_ms, fps, raw_poses = load_data(hdf5_dir)

    print(f"  Frames: {len(t_sys_ns)},  raw poses received: {raw_poses},  target fps: {fps}")

    m = compute_metrics(t_sys_ns, t_hw_ms, frame_no, ee_T, joints, offset_ms, fps)
    t_s = m["t_s"]
    ee_pos = m["ee_pos"]

    # Console summary
    border = "=" * 56
    print(f"\n{border}")
    print(f"  Temporal Alignment Verdict: {m['verdict']}")
    print(f"{border}")
    print(f"  Frames          : {m['N']}  ({m['duration_s']:.2f} s @ {m['actual_fps']:.1f} fps)")
    print(f"  Raw poses        : {raw_poses}")
    print(f"  Pose offset      : median={m['off_median']:.2f} ms  "
          f"p95={m['off_p95']:.2f} ms  max={m['off_max']:.2f} ms")
    print(f"  ½-frame threshold: {m['half_frame_ms']:.2f} ms")
    print(f"  Frame jitter     : median={m['dt_median']:.2f} ms  std={m['dt_std']:.2f} ms")
    print(f"  Dropped frames   : {m['dropped']} gaps  ({m['total_missing']} missing frames)")
    print(f"{border}\n")

    print(f"Rendering visualization → {out_path}")
    make_figure(
        t_s, offset_ms, m["dt_hw_ms"], ee_pos, m["ee_speed"], joints, m,
        fps, raw_poses, object_dir.name, out_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
