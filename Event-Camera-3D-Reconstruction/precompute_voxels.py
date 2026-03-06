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


def events_to_voxel_grid(
    events: np.ndarray,
    height: int,
    width: int,
    num_bins: int = 5,
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


def process_sequence(sequence_dir: Path, num_bins: int = 5, overwrite: bool = False) -> dict:
    """Process a single sequence directory."""
    sequence_dir = Path(sequence_dir)
    
    # Paths
    depth_h5_path = sequence_dir / "hdf5" / "depth.h5"
    events_path = sequence_dir / "events" / "events.npy"
    voxels_dir = sequence_dir / "events" / "voxels"
    
    result = {
        "name": sequence_dir.name,
        "success": False,
        "n_frames": 0,
        "error": None,
    }
    
    # Check input files
    if not depth_h5_path.exists():
        result["error"] = f"Depth HDF5 not found: {depth_h5_path}"
        return result
    if not events_path.exists():
        result["error"] = f"Events file not found: {events_path}"
        return result
    
    # Check if already processed
    if voxels_dir.exists() and not overwrite:
        existing = list(voxels_dir.glob("voxel_*.npy"))
        if len(existing) > 0:
            result["success"] = True
            result["n_frames"] = len(existing)
            result["error"] = "Already processed (use --overwrite to reprocess)"
            return result
    
    # Create output directory
    voxels_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load depth timestamps
        with h5py.File(depth_h5_path, 'r') as f:
            depth_shape = f['realsense/depth'].shape
            depth_timestamps = f['realsense/t_sys_ns'][:]
        
        n_frames = len(depth_timestamps)
        H, W = depth_shape[1], depth_shape[2]
        
        # Load all events
        events = np.load(events_path)
        n_events = len(events)
        
        # Sort events by timestamp for efficient slicing
        event_times = events['t']
        sort_idx = np.argsort(event_times)
        events = events[sort_idx]
        event_times = event_times[sort_idx]
        
        # Process each frame
        for frame_idx in range(n_frames):
            # Get time window
            t_start = depth_timestamps[frame_idx]
            if frame_idx < n_frames - 1:
                t_end = depth_timestamps[frame_idx + 1]
            else:
                t_end = event_times[-1] + 1 if len(event_times) > 0 else t_start + 1
            
            # Find events in time window using binary search
            idx_start = np.searchsorted(event_times, t_start, side='left')
            idx_end = np.searchsorted(event_times, t_end, side='left')
            
            frame_events = events[idx_start:idx_end]
            
            # Convert to voxel grid
            voxel = events_to_voxel_grid(frame_events, H, W, num_bins)
            
            # Save
            voxel_path = voxels_dir / f"voxel_{frame_idx:06d}.npy"
            np.save(voxel_path, voxel)
        
        result["success"] = True
        result["n_frames"] = n_frames
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find valid sequence directories."""
    sequence_dirs = []
    for d in data_root.iterdir():
        if d.is_dir():
            depth_h5 = d / "hdf5" / "depth.h5"
            events_npy = d / "events" / "events.npy"
            if depth_h5.exists() and events_npy.exists():
                sequence_dirs.append(d)
    return sorted(sequence_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute voxel grids from raw events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", nargs="+", type=str, default=None,
                       help="Sequence directory(ies) to process")
    parser.add_argument("--data_root", type=str, default="data/synthetic_data",
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
