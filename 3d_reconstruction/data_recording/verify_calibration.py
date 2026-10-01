#!/usr/bin/env python3
"""Verify and visualize multi-camera calibration results.

Loads all calibration files from camera_data/, re-runs ChArUco detection on
the saved images, and reports / visualises:

  1. Summary table   – all intrinsics and extrinsics
  2. Per-image reprojection errors for RGB and event cameras
  3. Image overlays   – detected vs re-projected corners
  4. 3D camera-frame plot – positions/orientations of all cameras

Usage:
    python data_recording/verify_calibration.py [--data-dir camera_data] [--no-plots]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from helpers import create_charuco_board as _make_charuco, detect_charuco

# Try interactive backends in order; fall back to Agg (file-only) if none works
import matplotlib
_INTERACTIVE = False
for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "wxAgg"):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as plt
        plt.figure()   # triggers display connection
        plt.close()
        _INTERACTIVE = True
        break
    except Exception:
        pass
if not _INTERACTIVE:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("[verify] No display found – plots will be saved to files instead.")

matplotlib.rcParams["figure.dpi"] = 120

# ── ChArUco board (must match recording) ────────────────────────────
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_H = 6
SQUARES_V = 9
SQUARE_LEN = 0.03
MARKER_LEN = 0.015


# ════════════════════════════════════════════════════════════════════
#  ChArUco helpers
# ════════════════════════════════════════════════════════════════════
def _detect(image, detector):
    return detect_charuco(image, detector, min_corners=4)


def _load_images(folder: Path) -> list[tuple[str, np.ndarray]]:
    """Load sorted .png images, return list of (name, image)."""
    paths = sorted(folder.glob("*.png"))
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            out.append((p.name, img))
    return out


# ════════════════════════════════════════════════════════════════════
#  Reprojection error
# ════════════════════════════════════════════════════════════════════
def reprojection_errors(
    images: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    board,
    detector,
    label: str,
) -> tuple[list[float], list[np.ndarray]]:
    """Compute per-image reprojection errors.

    Returns (per_image_rms_list, rvec_list_for_detected_images).
    """
    errors = []
    rvecs = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        corners, ids = _detect(gray, detector)
        if corners is None:
            errors.append(None)
            rvecs.append(None)
            continue

        obj_pts_all = board.getChessboardCorners()
        obj_pts = obj_pts_all[ids.flatten()].astype(np.float32)
        img_pts = corners.reshape(-1, 1, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            errors.append(None)
            rvecs.append(None)
            continue

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2)))
        errors.append(err)
        rvecs.append((rvec, tvec))

    valid = [e for e in errors if e is not None]
    detected = sum(1 for e in errors if e is not None)
    print(
        f"  [{label}] {detected}/{len(images)} images detected  "
        + (f"mean={np.mean(valid):.3f}px  max={np.max(valid):.3f}px" if valid else "NO DETECTIONS")
    )
    return errors, rvecs


# ════════════════════════════════════════════════════════════════════
#  Image overlay plot
# ════════════════════════════════════════════════════════════════════
def _draw_overlay(img, K, dist, board, detector) -> np.ndarray | None:
    """Draw detected (green) and reprojected (red) corners on a copy of img."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if img.ndim != 3 else img.copy()

    corners, ids = _detect(gray, detector)
    if corners is None:
        return None

    # Draw detected corners in green
    for c in corners:
        cv2.circle(vis, tuple(c.flatten().astype(int)), 5, (0, 255, 0), -1)

    obj_pts_all = board.getChessboardCorners()
    obj_pts = obj_pts_all[ids.flatten()].astype(np.float32)
    img_pts = corners.reshape(-1, 1, 2).astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return vis

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    for p in proj.reshape(-1, 2):
        cv2.circle(vis, tuple(p.astype(int)), 3, (0, 0, 255), -1)

    # Compute per-image RMS
    err = np.sqrt(np.mean((proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2))
    cv2.putText(vis, f"RMS {err:.2f}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return vis


def plot_overlays(rgb_imgs, ev_imgs, K_rgb, d_rgb, K_ev, d_ev, board, det, max_cols=4):
    """Side-by-side RGB / event overlay grid."""
    n = min(len(rgb_imgs), len(ev_imgs))
    if n == 0:
        return

    rows = (n + max_cols - 1) // max_cols
    fig, axes = plt.subplots(rows * 2, max_cols, figsize=(4 * max_cols, 4 * rows * 2))
    axes = np.array(axes).reshape(rows * 2, max_cols)

    for i in range(n):
        row_rgb = (i // max_cols) * 2
        row_ev = row_rgb + 1
        col = i % max_cols

        vis_rgb = _draw_overlay(rgb_imgs[i], K_rgb, d_rgb, board, det)
        vis_ev = _draw_overlay(ev_imgs[i], K_ev, d_ev, board, det)

        def _show(ax, vis, title):
            ax.set_title(title, fontsize=7)
            ax.axis("off")
            if vis is not None:
                ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB) if vis.ndim == 3 else vis, cmap="gray")
            else:
                ax.text(0.5, 0.5, "no detection", ha="center", va="center",
                        transform=ax.transAxes, color="red")

        _show(axes[row_rgb, col], vis_rgb, f"RGB #{i}")
        _show(axes[row_ev, col], vis_ev, f"Event #{i}")

    # Hide unused axes
    for i in range(n, rows * max_cols):
        axes[(i // max_cols) * 2, i % max_cols].axis("off")
        axes[(i // max_cols) * 2 + 1, i % max_cols].axis("off")

    fig.suptitle("Corner overlays  |  Green=detected  Red=reprojected", fontsize=10)
    plt.tight_layout()


# ════════════════════════════════════════════════════════════════════
#  Reprojection error bar chart
# ════════════════════════════════════════════════════════════════════
def plot_error_bars(rgb_errors, ev_errors):
    n = max(len(rgb_errors), len(ev_errors))
    idxs = np.arange(n)
    width = 0.35

    rgb_vals = [e if e is not None else 0.0 for e in rgb_errors]
    ev_vals  = [e if e is not None else 0.0 for e in ev_errors]
    rgb_miss = [e is None for e in rgb_errors]
    ev_miss  = [e is None for e in ev_errors]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), 4))
    b1 = ax.bar(idxs - width / 2, rgb_vals[:n], width, label="RGB", color="steelblue")
    b2 = ax.bar(idxs + width / 2, ev_vals[:n],  width, label="Event", color="darkorange")

    # Mark missed detections
    for i, m in enumerate(rgb_miss[:n]):
        if m:
            ax.text(i - width / 2, 0.02, "✗", ha="center", color="steelblue", fontsize=10)
    for i, m in enumerate(ev_miss[:n]):
        if m:
            ax.text(i + width / 2, 0.02, "✗", ha="center", color="darkorange", fontsize=10)

    ax.set_xlabel("Pose index")
    ax.set_ylabel("Reprojection error (px)")
    ax.set_title("Per-pose reprojection errors (✗ = no detection)")
    ax.set_xticks(idxs)
    ax.legend()
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="1 px threshold")
    plt.tight_layout()


# ════════════════════════════════════════════════════════════════════
#  3-D camera layout
# ════════════════════════════════════════════════════════════════════
def _T_to_pos_axes(T):
    """Extract position and axis directions from a 4x4 transform."""
    pos = T[:3, 3]
    R = T[:3, :3]
    return pos, R[:, 0], R[:, 1], R[:, 2]


def plot_camera_layout(T_color_from_depth, T_event_from_rgb, T_event_from_depth,
                       T_rgb_from_ee=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Depth camera at origin
    T_depth = np.eye(4)
    # RGB camera (from depth)
    T_rgb = T_color_from_depth
    # Event camera (from depth)
    T_ev = T_event_from_depth

    cameras = [
        ("Depth",  T_depth, "blue"),
        ("RGB",    T_rgb,   "green"),
        ("Event",  T_ev,    "red"),
    ]
    if T_rgb_from_ee is not None:
        T_ee_raw = np.linalg.inv(T_rgb_from_ee) @ T_color_from_depth
        cameras.append(("EE", T_ee_raw, "purple"))

    # Remap all camera T matrices into the reference frame (EE if available,
    # otherwise depth).  A single matrix-multiply moves every camera pose into
    # the chosen frame, so positions AND axis-orientations in the plot are all
    # consistent with what is printed.
    if T_rgb_from_ee is not None:
        # T_depth_from_ee maps depth-frame coords into EE-frame coords.
        T_depth_from_ee = np.linalg.inv(T_color_from_depth) @ T_rgb_from_ee
        cameras = [(name, T @ T_depth_from_ee, color) for name, T, color in cameras]
        ref = "EE"
    else:
        ref = "depth"

    # Now T[:3, 3] of every camera entry is its position in the ref frame.
    positions = {name: T[:3, 3] for name, T, _ in cameras}
    print(f"\n[verify] Camera positions (used in plot 03) — relative to {ref}:")
    for name, pos in positions.items():
        pcm = pos * 100
        print(f"  {name}: [{pcm[0]:+.2f}, {pcm[1]:+.2f}, {pcm[2]:+.2f}] cm")

    length = 0.02
    for name, T, color in cameras:
        pos, x_ax, y_ax, z_ax = _T_to_pos_axes(T)
        ax.scatter(*pos, s=80, color=color, label=name, zorder=5)
        ax.text(pos[0], pos[1], pos[2], f" {name}", fontsize=9, color=color)
        for vec, c in [(x_ax, "r"), (y_ax, "g"), (z_ax, "b")]:
            ax.quiver(*pos, *vec, length=length, normalize=True,
                      color=c, alpha=0.7, linewidth=1.5)

    # Draw lines between cameras (positions dict already built above)
    fixed_pairs = [("Depth", "RGB"), ("RGB", "Event"), ("Depth", "Event")]
    ee_pairs    = [("RGB", "EE")] if "EE" in positions else []
    for (a, b) in fixed_pairs + ee_pairs:
        if a not in positions or b not in positions:
            continue
        p0, p1 = positions[a], positions[b]
        dist = np.linalg.norm(p1 - p0) * 100  # cm
        mid = (p0 + p1) / 2
        ax.plot(*zip(p0, p1), "k--", alpha=0.4, linewidth=1)
        ax.text(mid[0], mid[1], mid[2], f"{dist:.1f} cm", fontsize=7, color="gray")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Camera layout (axes: R=X, G=Y, B=Z)\nRelative to {ref} frame")
    ax.legend()

    # Auto-range
    all_pos = np.array([T[:3, 3] for _, T, _ in cameras])
    span = np.ptp(all_pos, axis=0).max()
    span = max(span, 0.05)
    ctr = all_pos.mean(axis=0)
    half = span * 0.8
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)

    plt.tight_layout()


def plot_camera_layout_xy(T_color_from_depth, T_event_from_rgb, T_event_from_depth,
                          T_rgb_from_ee=None):
    """2-D top-down (X-Y plane) view of all camera positions, matching plot 03."""
    fig, ax = plt.subplots(figsize=(7, 6))

    T_depth = np.eye(4)
    T_rgb   = T_color_from_depth
    T_ev    = T_event_from_depth
    cameras = [
        ("Depth", T_depth, "blue"),
        ("RGB",   T_rgb,   "green"),
        ("Event", T_ev,    "red"),
    ]
    if T_rgb_from_ee is not None:
        T_ee_raw = np.linalg.inv(T_rgb_from_ee) @ T_color_from_depth
        cameras.append(("EE", T_ee_raw, "purple"))

    # Remap into EE frame (same logic as plot 03)
    if T_rgb_from_ee is not None:
        T_depth_from_ee = np.linalg.inv(T_color_from_depth) @ T_rgb_from_ee
        cameras = [(name, T @ T_depth_from_ee, color) for name, T, color in cameras]
        ref = "EE"
    else:
        ref = "depth"

    length = 0.015  # axis arrow length in metres

    positions = {}
    for name, T, color in cameras:
        pos = T[:3, 3]
        R   = T[:3, :3]
        x, y = pos[0], pos[1]
        positions[name] = np.array([x, y])

        ax.scatter(x, y, s=100, color=color, zorder=5)
        ax.annotate(f" {name}", (x, y), fontsize=9, color=color,
                    va="center", ha="left")

        # Draw X (red) and Y (green) axes projected onto XY plane
        for vec, c in [(R[:, 0], "red"), (R[:, 1], "green")]:
            dx, dy = vec[0] * length, vec[1] * length
            ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.5))

    # Draw dashed lines between cameras
    fixed_pairs = [("Depth", "RGB"), ("RGB", "Event"), ("Depth", "Event")]
    ee_pairs    = [("RGB", "EE")] if "EE" in positions else []
    for (a, b) in fixed_pairs + ee_pairs:
        if a not in positions or b not in positions:
            continue
        p0, p1 = positions[a], positions[b]
        dist = np.linalg.norm(p1 - p0) * 100  # cm
        mid  = (p0 + p1) / 2
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "k--", alpha=0.4, linewidth=1)
        ax.text(mid[0], mid[1], f"{dist:.1f} cm", fontsize=7, color="gray",
                ha="center", va="bottom")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Camera layout — X/Y plane (top-down)\nRelative to {ref} frame  |  red=X  green=Y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Auto-range
    all_xy = np.array(list(positions.values()))
    span = np.ptp(all_xy, axis=0).max()
    span = max(span, 0.05)
    ctr  = all_xy.mean(axis=0)
    half = span * 0.8
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)

    plt.tight_layout()


# ════════════════════════════════════════════════════════════════════
#  Projection check  (board plane from one camera projected into the other)
# ════════════════════════════════════════════════════════════════════
def _project_and_draw(src_img, tgt_img, src_K, src_dist, tgt_K, tgt_dist,
                      T_tgt_from_src, board, det):
    """Project all board corners from src into tgt.

    Returns a BGR vis image with:
      - red dots + outline  = projected corners (from src via extrinsics)
      - green dots          = corners actually detected in tgt
      - yellow cross-cam RMS text when both cameras found common corners
    Returns None when the board is not detected in src.
    """
    scorners, sids = _detect(src_img, det)
    if scorners is None:
        return None

    obj_pts = board.getChessboardCorners()[sids.flatten()].astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, scorners.reshape(-1, 1, 2).astype(np.float32), src_K, src_dist
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)

    # All board corners → source camera frame → target camera frame
    obj_all = board.getChessboardCorners().astype(np.float64)
    pts_src = (R @ obj_all.T + t).T
    pts_tgt = (T_tgt_from_src @ np.hstack([pts_src, np.ones((len(pts_src), 1))]).T).T[:, :3]

    proj, _ = cv2.projectPoints(
        pts_tgt.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), tgt_K, tgt_dist
    )
    proj = proj.reshape(-1, 2)

    tgt_bgr = tgt_img.copy() if tgt_img.ndim == 3 else cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR)
    vis = tgt_bgr.copy()

    # Projected board outline + dots (red in BGR = 50,50,220)
    cv2.polylines(vis, [proj.astype(int).reshape(-1, 1, 2)],
                  isClosed=True, color=(50, 50, 220), thickness=2)
    for p in proj.astype(int):
        cv2.circle(vis, tuple(p.tolist()), 3, (50, 50, 220), -1)

    # Detected corners in target (green)
    tcorners, tids = _detect(tgt_img, det)
    if tcorners is not None:
        for c in tcorners:
            cv2.circle(vis, tuple(c.flatten().astype(int)), 5, (0, 200, 0), -1)

        # Cross-camera reprojection error for common corner IDs
        common = np.intersect1d(sids.flatten(), tids.flatten())
        if len(common) >= 4:
            proj_common = proj[common]          # proj indexed by global corner id
            tm = np.isin(tids.flatten(), common)
            t_order = np.argsort(tids.flatten()[tm])
            tgt_common = tcorners[tm][t_order].reshape(-1, 2)
            err = float(np.sqrt(np.mean((proj_common - tgt_common) ** 2)))
            cv2.putText(vis, f"cross-cam RMS {err:.2f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Legend
    h, w = vis.shape[:2]
    cv2.putText(vis, "red=projected  green=detected", (w - 290, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return vis


def run_projection_check(rgb_imgs, ev_imgs, K_rgb, d_rgb, K_ev, d_ev,
                         T_ev_from_rgb, board, det, out_dir: Path):
    """Project board corners between cameras for all poses; save to out_dir."""
    T_rgb_from_ev = np.linalg.inv(T_ev_from_rgb)
    saved = 0
    for i, (rgb, ev) in enumerate(zip(rgb_imgs, ev_imgs)):
        vis_re = _project_and_draw(rgb, ev, K_rgb, d_rgb, K_ev, d_ev,
                                   T_ev_from_rgb, board, det)
        if vis_re is not None:
            cv2.imwrite(str(out_dir / f"proj_rgb_to_event_{i:03d}.png"), vis_re)
            saved += 1

        vis_er = _project_and_draw(ev, rgb, K_ev, d_ev, K_rgb, d_rgb,
                                   T_rgb_from_ev, board, det)
        if vis_er is not None:
            cv2.imwrite(str(out_dir / f"proj_event_to_rgb_{i:03d}.png"), vis_er)
            saved += 1

    print(f"[verify] Projection check: {saved} images saved → {out_dir}")


# ════════════════════════════════════════════════════════════════════
#  Summary
# ════════════════════════════════════════════════════════════════════
def _fmt_K(K, name):
    print(f"\n  {name}:")
    print(f"    fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")


def print_summary(data_dir: Path):
    print("\n" + "═" * 60)
    print("  CALIBRATION SUMMARY")
    print("═" * 60)

    # Intrinsics
    for fname, label in [
        ("rs_depth_intrinsics.npz", "Depth  (SDK)"),
        ("rs_rgb_intrinsics.npz",   "RGB    (ChArUco)"),
        ("event_intrinsics.npz",    "Event  (ChArUco)"),
    ]:
        p = data_dir / fname
        if not p.exists():
            print(f"\n  {label}: MISSING ({fname})")
            continue
        d = np.load(str(p))
        K = d["camera_matrix"]
        dist = d["dist_coeffs"]
        rms = float(d["rms"]) if "rms" in d else None
        sz = tuple(d["image_size"]) if "image_size" in d else "?"
        _fmt_K(K, label)
        print(f"    dist={np.array2string(dist, precision=4, suppress_small=True)}")
        print(f"    image_size={sz}" + (f"  calibration_RMS={rms:.4f}px" if rms else ""))

    # Extrinsics
    print("\n" + "─" * 60)
    for fname, label in [
        ("T_color_from_depth.npz",  "T  depth -> RGB     (SDK)"),
        ("T_event_from_rgb.npz",    "T  RGB   -> Event   (stereo)"),
        ("T_event_from_depth.npz",  "T  depth -> Event   (composed)"),
        ("T_rgb_from_ee.npz",       "T  EE    -> RGB     (hand-eye)"),
    ]:
        p = data_dir / fname
        if not p.exists():
            print(f"\n  {label}: MISSING")
            continue
        T = np.load(str(p))["T"]
        t = T[:3, 3] * 100  # in cm
        euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        print(f"\n  {label}")
        print(f"    translation: [{t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}] cm")
        print(f"    rotation XYZ: [{euler[0]:+.2f}, {euler[1]:+.2f}, {euler[2]:+.2f}] deg")

    depth_scale_path = data_dir / "depth_scale.npz"
    if depth_scale_path.exists():
        scale = float(np.load(str(depth_scale_path))["scale"])
        print(f"\n  Depth scale: {scale:.6f} m/unit")
    print("\n" + "═" * 60)


# ════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Verify and visualize multi-camera calibration results"
    )
    parser.add_argument(
        "--data-dir", default=_PROJECT_DIR / "camera_data",
        help="Directory with calibration files (default: camera_data)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Print summary only, skip all plots"
    )
    parser.add_argument(
        "--max-cols", type=int, default=4,
        help="Max columns in overlay grid (default: 4)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = _PROJECT_DIR / data_dir
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # 1. Summary
    print_summary(data_dir)

    # Load calibration files
    def _need(fname):
        p = data_dir / fname
        if not p.exists():
            print(f"[verify] Missing {fname} – skipping dependent checks")
            return None
        return np.load(str(p))

    rgb_cal   = _need("rs_rgb_intrinsics.npz")
    ev_cal    = _need("event_intrinsics.npz")
    dep_cal   = _need("T_color_from_depth.npz")
    ev_rgb    = _need("T_event_from_rgb.npz")
    ev_depth  = _need("T_event_from_depth.npz")
    # T_rgb_from_ee is optional – present only when eye-in-hand was run
    _ee_path  = data_dir / "T_rgb_from_ee.npz"
    ee_cal    = np.load(str(_ee_path)) if _ee_path.exists() else None

    if args.no_plots:
        sys.exit(0)

    board, det = _make_charuco()

    # 2. Load images
    rgb_dir = data_dir / "rgb_frames"
    ev_dir  = data_dir / "event_frames"

    rgb_pairs = _load_images(rgb_dir) if rgb_dir.exists() else []
    ev_pairs  = _load_images(ev_dir)  if ev_dir.exists()  else []
    rgb_imgs  = [img for _, img in rgb_pairs]
    ev_imgs   = [img for _, img in ev_pairs]

    print(f"\n[verify] RGB images:   {len(rgb_imgs)} from {rgb_dir}")
    print(f"[verify] Event images: {len(ev_imgs)} from {ev_dir}")

    # 3. Reprojection errors
    print("\n── Reprojection errors ──────────────────────────────────")
    rgb_errors, _ = reprojection_errors(
        rgb_imgs,
        rgb_cal["camera_matrix"], rgb_cal["dist_coeffs"],
        board, det, "RGB",
    ) if rgb_imgs and rgb_cal is not None else ([], [])

    ev_errors, _ = reprojection_errors(
        ev_imgs,
        ev_cal["camera_matrix"], ev_cal["dist_coeffs"],
        board, det, "Event",
    ) if ev_imgs and ev_cal is not None else ([], [])

    if args.no_plots:
        return

    # 4. Error bar chart
    if rgb_errors or ev_errors:
        plot_error_bars(
            rgb_errors if rgb_errors else [None] * len(ev_errors),
            ev_errors  if ev_errors  else [None] * len(rgb_errors),
        )

    # 5. Overlay images
    if rgb_imgs and ev_imgs and rgb_cal is not None and ev_cal is not None:
        n = min(len(rgb_imgs), len(ev_imgs))
        plot_overlays(
            rgb_imgs[:n], ev_imgs[:n],
            rgb_cal["camera_matrix"], rgb_cal["dist_coeffs"],
            ev_cal["camera_matrix"],  ev_cal["dist_coeffs"],
            board, det, max_cols=args.max_cols,
        )

    # 6. 3D camera layout
    if dep_cal is not None and ev_depth is not None and ev_rgb is not None:
        plot_camera_layout(
            T_color_from_depth=dep_cal["T"],
            T_event_from_rgb=ev_rgb["T"],
            T_event_from_depth=ev_depth["T"],
            T_rgb_from_ee=ee_cal["T"] if ee_cal is not None else None,
        )

    # 7. Top-down X/Y camera layout
    if dep_cal is not None and ev_depth is not None and ev_rgb is not None:
        plot_camera_layout_xy(
            T_color_from_depth=dep_cal["T"],
            T_event_from_rgb=ev_rgb["T"],
            T_event_from_depth=ev_depth["T"],
            T_rgb_from_ee=ee_cal["T"] if ee_cal is not None else None,
        )

    if _INTERACTIVE:
        plt.show()
    # Always save plots + projection check to calibration_visualization/
    vis_dir = data_dir / "calibration_visualization"
    vis_dir.mkdir(exist_ok=True)
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        p = vis_dir / f"plot_{i+1:02d}.png"
        fig.savefig(str(p), bbox_inches="tight")
        print(f"[verify] Saved {p}")
    plt.close("all")

    # 7. Projection check (RGB ↔ Event)
    if rgb_imgs and ev_imgs and rgb_cal is not None and ev_cal is not None and ev_rgb is not None:
        run_projection_check(
            rgb_imgs, ev_imgs,
            rgb_cal["camera_matrix"], rgb_cal["dist_coeffs"],
            ev_cal["camera_matrix"],  ev_cal["dist_coeffs"],
            ev_rgb["T"], board, det, vis_dir,
        )
    print(f"\n[verify] All outputs saved to {vis_dir}")


if __name__ == "__main__":
    main()
