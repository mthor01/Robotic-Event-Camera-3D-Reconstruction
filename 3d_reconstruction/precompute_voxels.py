#!/usr/bin/env python3
"""
Precompute voxel grids from raw events for faster training.

This script converts raw event data (events.npy) into per-frame voxel grids,
which dramatically reduces memory usage and speeds up training.

Usage:
    python precompute_voxels.py --data_root data/synthetic_data
    python precompute_voxels.py --data_dir data/synthetic_data/bottle data/synthetic_data/cube_medium
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from reconstruction_config import NUM_BINS


def events_to_voxel_grid(
    events: np.ndarray,
    height: int,
    width: int,
    num_bins: int = NUM_BINS,
) -> np.ndarray:
    """
    Convert raw events to a voxel grid representation.
    
    Args:
        events: Structured array with fields (x, y, p, t)
        height: Image height
        width: Image width
        num_bins: Number of temporal bins
        
    Returns:
        Voxel grid of shape (num_bins, height, width)
    """
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    
    if len(events) == 0:
        return voxel
    
    # Extract event fields
    x = events['x'].astype(np.int32)
    y = events['y'].astype(np.int32)
    p = events['p'].astype(np.float32)  # 0 or 1
    t = events['t'].astype(np.float64)
    
    # Apply 180° rotation to match the HDF5 event frames
    # (generate_event_videos applies cv2.ROTATE_180)
    x = (width  - 1) - x
    y = (height - 1) - y
    
    # Convert polarity: 0 -> -1, 1 -> +1
    p = p * 2 - 1
    
    # Normalize timestamps to [0, num_bins-1]
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)
    else:
        t_norm = np.zeros_like(t)
    
    # Bilinear interpolation across time bins
    t_floor = np.floor(t_norm).astype(np.int32)
    t_ceil = np.minimum(t_floor + 1, num_bins - 1)
    t_frac = t_norm - t_floor
    
    # Accumulate events into voxel grid
    for i in range(len(events)):
        if 0 <= x[i] < width and 0 <= y[i] < height:
            voxel[t_floor[i], y[i], x[i]] += p[i] * (1 - t_frac[i])
            voxel[t_ceil[i], y[i], x[i]] += p[i] * t_frac[i]
    
    # Normalize voxel grid
    nonzero_mask = voxel != 0
    if nonzero_mask.any():
        mean = voxel[nonzero_mask].mean()
        std = voxel[nonzero_mask].std()
        if std > 0:
            voxel = (voxel - mean) / std
    
    return voxel


def load_events_from_raw(raw_path: Path) -> np.ndarray:
    """
    Load all events from a Metavision .raw file into a sorted structured array.

    Returns a numpy structured array with fields (x, y, p, t) where t is in
    microseconds (event-camera internal clock, origin = device open time).
    """
    from metavision_core.event_io import EventsIterator

    chunks = []
    for ev_batch in EventsIterator(str(raw_path), delta_t=1_000_000):  # 1 s chunks
        if len(ev_batch) > 0:
            chunks.append(ev_batch.copy())

    if not chunks:
        return np.array([], dtype=np.dtype([
            ('x', '<u2'), ('y', '<u2'), ('p', 'u1'), ('t', '<i8')
        ]))

    events = np.concatenate(chunks)
    sort_idx = np.argsort(events['t'], kind='stable')
    return events[sort_idx]


def process_sequence(sequence_dir: Path, num_bins: int = 5, overwrite: bool = False) -> dict:
    """
    Process a single sequence directory.

    Reads depth timestamps from hdf5/realsense.h5 and raw events from
    raw/event_cam*.raw.  For each depth frame i the events in the half-open
    interval [t_depth[i], t_depth[i+1]) are accumulated into a voxel grid
    with `num_bins` temporal bins and saved to
    events/voxels_cam{k}/voxel_{i:06d}.npy.
    """
    sequence_dir = Path(sequence_dir)

    realsense_h5_path = sequence_dir / "hdf5" / "realsense.h5"
    raw_dir           = sequence_dir / "raw_event_data"

    result = {
        "name": sequence_dir.name,
        "success": False,
        "n_frames": 0,
        "error": None,
    }

    if not realsense_h5_path.exists():
        result["error"] = f"realsense.h5 not found: {realsense_h5_path}"
        return result

    # Discover raw event files (events_cam0.raw, events_cam1.raw, ...)
    raw_event_files = sorted(raw_dir.glob("events_cam*.raw")) if raw_dir.exists() else []
    if not raw_event_files:
        result["error"] = f"No raw event files found in {raw_dir}"
        return result

    # Load depth shape and system-clock timestamps (nanoseconds)
    with h5py.File(realsense_h5_path, 'r') as f:
        depth_shape = f['depth'].shape   # (N, H, W)
        depth_t_ns  = f['t_sys_ns'][:]   # (N,) nanoseconds

    n_frames = len(depth_t_ns)

    # Event camera native resolution (read from the aligned HDF5 file)
    # We must NOT use the RealSense depth resolution here — raw event
    # x/y are in the event camera's own coordinate system.
    ev_h5_path = sequence_dir / "hdf5" / "events_cam0.h5"
    if ev_h5_path.exists():
        with h5py.File(ev_h5_path, 'r') as f:
            EV_H = int(f["events"].attrs["height"])
            EV_W = int(f["events"].attrs["width"])
    else:
        # Fallback: read from raw file via EventsIterator
        from metavision_core.event_io import EventsIterator as _EI
        _it = _EI(str(raw_event_files[0]), delta_t=1_000_000)
        EV_H, EV_W = _it.get_size()
        del _it

    # Fast-path: all cameras already processed
    if not overwrite:
        all_done = all(
            len(list((sequence_dir / "events" / f"voxels_cam{i}").glob("voxel_*.npy"))) >= n_frames
            for i in range(len(raw_event_files))
        )
        if all_done:
            result["success"] = True
            result["n_frames"] = n_frames
            result["error"] = "Already processed (use --overwrite to reprocess)"
            return result

    # Time alignment strategy:
    #   The aligned events_cam{k}.h5 already has per-frame timestamps
    #   (t_ev_start_us, t_ev_end_us) in event-camera internal µs — the
    #   exact same clock domain as the raw .raw file.  We read those
    #   timestamps and use them directly to slice the raw event stream.
    #   This avoids any fragile cross-clock (system ↔ event-camera)
    #   conversion entirely.

    try:
        for cam_idx, raw_path in enumerate(raw_event_files):
            voxels_dir = sequence_dir / "events" / f"voxels_cam{cam_idx}"

            # Skip if this camera is already done
            if not overwrite and voxels_dir.exists():
                if len(list(voxels_dir.glob("voxel_*.npy"))) >= n_frames:
                    continue

            # Read per-frame time windows from the aligned HDF5
            aligned_h5 = sequence_dir / "hdf5" / f"events_cam{cam_idx}.h5"
            if not aligned_h5.exists():
                print(f"  [cam{cam_idx}] aligned HDF5 not found: {aligned_h5}")
                continue
            with h5py.File(aligned_h5, 'r') as f:
                ev_t_start_us = f['events/t_ev_start_us'][:]  # (N,) int64
                ev_t_end_us   = f['events/t_ev_end_us'][:]    # (N,) int64

            if len(ev_t_start_us) != n_frames:
                print(f"  [cam{cam_idx}] WARNING: HDF5 has {len(ev_t_start_us)} frames "
                      f"but depth has {n_frames}")

            voxels_dir.mkdir(parents=True, exist_ok=True)

            # Load all raw events for this camera into memory
            events = load_events_from_raw(raw_path)
            raw_t = events['t'].astype(np.int64)

            for frame_idx in tqdm(range(n_frames), desc=f"  cam{cam_idx}", leave=False):
                t_start = int(ev_t_start_us[frame_idx])
                t_end   = int(ev_t_end_us[frame_idx])

                i0 = np.searchsorted(raw_t, t_start, side='left')
                i1 = np.searchsorted(raw_t, t_end,   side='right')

                voxel = events_to_voxel_grid(events[i0:i1], EV_H, EV_W, num_bins)
                np.save(voxels_dir / f"voxel_{frame_idx:06d}.npy", voxel)

        result["success"] = True
        result["n_frames"] = n_frames

    except Exception as e:
        result["error"] = str(e)

    return result


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find valid sequence directories that contain the new data structure."""
    sequence_dirs = []
    for d in data_root.iterdir():
        if d.is_dir():
            realsense_h5 = d / "hdf5" / "realsense.h5"
            raw_dir = d / "raw_event_data"
            has_raw = raw_dir.exists() and len(list(raw_dir.glob("events_cam*.raw"))) > 0
            if realsense_h5.exists() and has_raw:
                sequence_dirs.append(d)
    return sorted(sequence_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute voxel grids from raw events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", nargs="+", type=str, default=None,
                       help="Sequence directory(ies) to process")
    parser.add_argument("--data_root", type=str, default="data/real",
                       help="Root data directory (will process all subdirs)")
    parser.add_argument("--num_bins", type=int, default=5,
                       help="Number of temporal bins for voxel grid")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing voxel files")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    
    # Find sequences
    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
    
    if not sequence_dirs:
        print(f"No valid sequences found!")
        return
    
    print(f"Found {len(sequence_dirs)} sequences to process")
    print(f"Voxel bins: {args.num_bins}")
    print(f"Overwrite: {args.overwrite}")
    print()
    
    # Process sequences
    n_workers = args.workers or min(multiprocessing.cpu_count(), len(sequence_dirs))
    
    if n_workers == 1 or len(sequence_dirs) == 1:
        # Sequential processing with progress bar
        results = []
        for seq_dir in tqdm(sequence_dirs, desc="Processing"):
            result = process_sequence(seq_dir, args.num_bins, args.overwrite)
            results.append(result)
            if result["success"]:
                tqdm.write(f"  ✓ {result['name']}: {result['n_frames']} frames")
            else:
                tqdm.write(f"  ✗ {result['name']}: {result['error']}")
    else:
        # Parallel processing
        print(f"Using {n_workers} workers")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_sequence, seq_dir, args.num_bins, args.overwrite): seq_dir
                for seq_dir in sequence_dirs
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                results.append(result)
                if result["success"]:
                    tqdm.write(f"  ✓ {result['name']}: {result['n_frames']} frames")
                else:
                    tqdm.write(f"  ✗ {result['name']}: {result['error']}")
    
    # Summary
    print()
    print("=" * 60)
    n_success = sum(1 for r in results if r["success"])
    n_total_frames = sum(r["n_frames"] for r in results if r["success"])
    print(f"Processed: {n_success}/{len(results)} sequences")
    print(f"Total frames: {n_total_frames}")
    
    # Show failures
    failures = [r for r in results if not r["success"] and "Already" not in str(r["error"])]
    if failures:
        print(f"\nFailures:")
        for r in failures:
            print(f"  - {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
