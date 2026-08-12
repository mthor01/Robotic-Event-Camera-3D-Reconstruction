#!/usr/bin/env python3
"""Render all frame-indexed HDF5 image modalities in one labeled row.

The first panel is an event image generated from the stored event voxel grid.
Remaining panels are discovered automatically from HDF5 files in ``hdf5/``
and ``events/``. Pose arrays, timestamps, and other non-image datasets are
ignored.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from config import DATA_ROOT, DEPTH_VIZ_MAX, DEPTH_VIZ_MIN  # noqa: E402


@dataclass
class Panel:
    title: str
    image: np.ndarray
    cmap: str | None = None
    vmin: float | None = None
    vmax: float | None = None


def resolve_sequence(sequence: str, data_root: Path) -> Path:
    supplied = Path(sequence).expanduser()
    if supplied.is_dir():
        return supplied.resolve()
    root = data_root.expanduser().resolve()
    direct = root / sequence
    if direct.is_dir():
        return direct
    matches = sorted(path for path in root.glob(f"*/{sequence}") if path.is_dir())
    if not matches:
        raise FileNotFoundError(
            f"Could not find sequence '{sequence}' as a path or below {root}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Sequence '{sequence}' is ambiguous: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def normalize_event_image(voxel: np.ndarray) -> np.ndarray:
    """Collapse temporal bins and color negative/positive events blue/red."""
    signed = np.asarray(voxel, dtype=np.float32).sum(axis=0)
    scale = float(np.percentile(np.abs(signed), 99.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = np.clip(signed / scale, -1.0, 1.0)
    image = np.ones((*signed.shape, 3), dtype=np.float32)
    positive = normalized > 0.0
    negative = normalized < 0.0
    strength = np.abs(normalized)
    image[positive, 0] = 1.0
    image[positive, 1] = 1.0 - strength[positive]
    image[positive, 2] = 1.0 - strength[positive]
    image[negative, 0] = 1.0 - strength[negative]
    image[negative, 1] = 1.0 - strength[negative]
    image[negative, 2] = 1.0
    return image


def is_image_sample(sample: np.ndarray) -> bool:
    if sample.ndim == 2:
        return min(sample.shape) >= 16
    if sample.ndim == 3:
        return sample.shape[-1] in (1, 3, 4) and min(sample.shape[:2]) >= 16
    return False


def panel_from_sample(title: str, sample: np.ndarray, depth_scale: float) -> Panel:
    image = np.asarray(sample)
    lower_title = title.lower()
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        return Panel(title, image)
    if "depth" in lower_title:
        depth = image.astype(np.float32)
        if "realsense.h5" in lower_title:
            depth *= depth_scale
        depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0.0), depth)
        return Panel(title, depth, "turbo", DEPTH_VIZ_MIN, DEPTH_VIZ_MAX)
    if "mask" in lower_title:
        return Panel(title, image, "gray", 0.0, 1.0)
    if "table_plane" in lower_title:
        return Panel(title, image, "viridis", 0.0, 1.0)
    finite = image[np.isfinite(image)]
    if finite.size:
        low, high = np.percentile(finite, (1.0, 99.0))
        if high <= low:
            high = low + 1.0
    else:
        low, high = 0.0, 1.0
    return Panel(title, image, "gray", float(low), float(high))


def discover_image_panels(
    sequence_dir: Path,
    frame: int,
    depth_scale: float,
) -> list[Panel]:
    panels: list[Panel] = []
    search_paths = sorted((sequence_dir / "hdf5").glob("*.h5"))
    search_paths += sorted((sequence_dir / "events").glob("*.h5"))
    for file_path in search_paths:
        with h5py.File(file_path, "r") as handle:
            datasets: list[tuple[str, h5py.Dataset]] = []

            def collect(name: str, item: object) -> None:
                if isinstance(item, h5py.Dataset):
                    datasets.append((name, item))

            handle.visititems(collect)
            for dataset_name, dataset in datasets:
                # The voxel tensor is rendered once as the dedicated event panel.
                if file_path.name == "voxels_cam0.h5" and dataset_name == "voxels":
                    continue
                if dataset.ndim < 3 or dataset.shape[0] <= frame:
                    continue
                sample_shape = dataset.shape[1:]
                likely_image = (
                    len(sample_shape) == 2 and min(sample_shape) >= 16
                ) or (
                    len(sample_shape) == 3
                    and sample_shape[-1] in (1, 3, 4)
                    and min(sample_shape[:2]) >= 16
                )
                if not likely_image:
                    continue
                sample = dataset[frame]
                if not is_image_sample(sample):
                    continue
                title = f"{file_path.name}\n{dataset_name}"
                panels.append(panel_from_sample(title, sample, depth_scale))
    return panels


def visualize(args: argparse.Namespace) -> Path:
    sequence_dir = resolve_sequence(args.sequence, args.data_root)
    voxel_path = sequence_dir / "events" / "voxels_cam0.h5"
    if not voxel_path.is_file():
        raise FileNotFoundError(f"Missing event voxel file: {voxel_path}")
    with h5py.File(voxel_path, "r") as handle:
        if "voxels" not in handle:
            raise KeyError(f"{voxel_path} does not contain 'voxels'")
        if not 0 <= args.frame < len(handle["voxels"]):
            raise IndexError(
                f"Frame {args.frame} is outside the voxel range "
                f"0..{len(handle['voxels']) - 1}"
            )
        event_image = normalize_event_image(handle["voxels"][args.frame])

    depth_scale_path = args.calib_dir / "depth_scale.npz"
    depth_scale = (
        float(np.load(depth_scale_path)["scale"])
        if depth_scale_path.is_file()
        else 1.0
    )
    panels = [Panel("Events\nvoxel projection", event_image)]
    panels.extend(discover_image_panels(sequence_dir, args.frame, depth_scale))
    if len(panels) == 1:
        raise RuntimeError(f"No frame-indexed HDF5 image datasets found in {sequence_dir}")

    panel_width = args.panel_width
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(panel_width * len(panels), panel_width * 0.82),
        squeeze=False,
    )
    for axis, panel in zip(axes[0], panels):
        axis.imshow(
            panel.image,
            cmap=panel.cmap,
            vmin=panel.vmin,
            vmax=panel.vmax,
            interpolation="nearest",
        )
        axis.set_title(panel.title, fontsize=10, pad=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(
        f"{sequence_dir.name}, frame {args.frame}", fontsize=14, fontweight="bold"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=0.35)

    output = args.output
    if output is None:
        output = args.output_dir / (
            f"h5_modalities_{sequence_dir.name}_frame_{args.frame:05d}.png"
        )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Sequence: {sequence_dir}")
    print(f"Frame:    {args.frame}")
    print(f"Panels:   {len(panels)}")
    for panel in panels:
        print(f"  - {panel.title.replace(chr(10), ': ')}")
    print(f"Saved:    {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place the event image and all frame-indexed HDF5 images in one row."
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Sequence name below the data root, or a sequence-directory path.",
    )
    parser.add_argument("--frame", type=int, required=True, help="Zero-based frame index.")
    parser.add_argument(
        "--data-root", type=Path, default=_ROOT / DATA_ROOT,
        help="Dataset root searched when --sequence is a name.",
    )
    parser.add_argument(
        "--calib-dir", type=Path, default=_ROOT / "camera_data",
        help="Calibration directory containing depth_scale.npz.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_HERE / "plots",
        help="Output directory used when --output is omitted.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Explicit output PNG path.")
    parser.add_argument("--panel-width", type=float, default=3.2)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.frame < 0:
        parser.error("--frame must be non-negative")
    if args.panel_width <= 0.0:
        parser.error("--panel-width must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


if __name__ == "__main__":
    visualize(parse_args())
