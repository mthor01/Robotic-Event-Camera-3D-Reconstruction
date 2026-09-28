#!/usr/bin/env python3
"""
Precompute frame-aligned voxel grids from raw events.

Raw event-camera recordings are split by the RealSense hardware-trigger
timestamps and accumulated into temporal bins. Trigger data are read from the
RAW recording, with the event HDF5 copy used as a fallback. Each grid receives
the same resize and centred crop used by depth projection and model inference.
The resulting ``hdf5/voxels.h5`` can be loaded directly during training.

Usage:
    python3 data_precomputation/precompute_voxels.py --data_root data/new/train
    python3 data_precomputation/precompute_voxels.py --data_dir data/new/train/1 data/new/train/2

Hardware-trigger alignment is required; recordings without trigger timestamps
are rejected instead of falling back to cross-clock elapsed-time estimates.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from config import (
    DATA_ROOT as _DATA_ROOT, NUM_BINS, PREPROCESS_CROP_HW,
    PREPROCESS_RESIZE_HW,
)


DATA_ROOT = _HERE.parent / _DATA_ROOT


def events_to_voxel_grid(
    events: np.ndarray,
    height: int,
    width: int,
    num_bins: int = NUM_BINS,
    t_start_us: Optional[float] = None,
    t_end_us: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert raw events to a voxel grid representation.

    Bins are fixed-duration slices of [t_start_us, t_end_us] (the full frame
    time window) so that bin k covers a predictable interval regardless of
    whether events are sparse or dense.  When t_start_us / t_end_us are not
    provided the function falls back to data-dependent [t_min, t_max], which
    makes bin timing depend on actual event activity — avoid for alignment.

    Args:
        events:     Structured array with fields (x, y, p, t), t in µs.
        height:     Image height.
        width:      Image width.
        num_bins:   Number of temporal bins.
        t_start_us: Frame start time in µs (event-camera clock).
        t_end_us:   Frame end   time in µs (event-camera clock).
        normalize:  If True, standardise active voxel entries per frame.

    Returns:
        Voxel grid of shape (num_bins, height, width).
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
    
    # Normalize timestamps to [0, num_bins-1] using fixed frame boundaries so
    # every bin covers the same wall-clock duration (delta_t / num_bins).
    # Fall back to data range only when no frame boundaries are provided.
    if t_start_us is not None and t_end_us is not None and t_end_us > t_start_us:
        t_min = float(t_start_us)
        t_max = float(t_end_us)
    else:
        t_min, t_max = float(t.min()), float(t.max())
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
    
    # Normalize voxel grid over active entries only. Keep inactive pixels at
    # zero so empty background does not become dense nonzero signal.
    if normalize:
        nonzero_mask = voxel != 0
        if nonzero_mask.any():
            mean = voxel[nonzero_mask].mean()
            std = voxel[nonzero_mask].std()
            if std > 0:
                voxel[nonzero_mask] = (voxel[nonzero_mask] - mean) / std

    return voxel


def resize_voxel(
    voxel: np.ndarray,
    output_hw: Tuple[int, int],
    as_float16: bool = False,
) -> np.ndarray:
    """
    Downsample a (C, H, W) voxel grid to output_hw using bilinear interpolation.

    Args:
        voxel:      (C, H, W) float32 array.
        output_hw:  (out_H, out_W) target spatial size.
        as_float16: If True, cast the result to float16 before returning.

    Returns:
        (C, out_H, out_W) array.
    """
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(voxel).unsqueeze(0)  # (1, C, H, W)
    t = F.interpolate(t, size=output_hw, mode='bilinear', align_corners=False)
    out = t.squeeze(0).numpy()  # (C, out_H, out_W)
    if as_float16:
        out = out.astype(np.float16)
    return out


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


def process_sequence(
    sequence_dir: Path,
    num_bins: int = 5,
    overwrite: bool = False,
    output_hw: Tuple[int, int] = PREPROCESS_RESIZE_HW,
    as_float16: bool = False,
    crop_hw: Tuple[int, int] = PREPROCESS_CROP_HW,
    show_progress: bool = True,
    normalize: bool = True,
) -> dict:
    """
    Process a single sequence directory.

    Reads the frame count from hdf5/realsense.h5 and raw events from
    raw_event_data/events_cam*.raw. For every hardware trigger, events in a
    trigger-centred window are accumulated into ``num_bins`` temporal bins.
    Each voxel is center-cropped at native resolution and then resized. Use
    ``as_float16`` to reduce storage for normalized event data.

    Each per-frame window is centred on its hardware-trigger timestamp:
        window = [trigger[i] - half_period, trigger[i] + half_period)
    so the middle temporal bin of the voxel grid falls exactly on the trigger.

    Saved to events/voxels_cam{k}.h5 as HDF5 dataset "voxels" with shape
    (N, num_bins, H, W).
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

    # Load depth shape and frame count
    with h5py.File(realsense_h5_path, 'r') as f:
        depth_shape = f['depth'].shape   # (N, H, W)
        # t_global_ms is the RS2 global (system-clock) timestamp; older recordings
        # used t_sys_ns. We only need the array to determine n_frames here.
        if 't_global_ms' in f:
            n_frames = f['t_global_ms'].shape[0]
        else:
            n_frames = f['t_sys_ns'].shape[0]

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
        def _cam_done(i: int) -> bool:
            h5 = sequence_dir / "events" / f"voxels_cam{i}.h5"
            if h5.exists():
                with h5py.File(h5, 'r') as _f:
                    return int(_f["voxels"].shape[0]) >= n_frames
            return False
        if all(_cam_done(i) for i in range(len(raw_event_files))):
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
            voxels_h5 = sequence_dir / "events" / f"voxels_cam{cam_idx}.h5"

            # Skip if this camera is already done
            if not overwrite and voxels_h5.exists():
                with h5py.File(voxels_h5, 'r') as _f:
                    if int(_f["voxels"].shape[0]) >= n_frames:
                        continue

            # Read per-frame time windows from the aligned HDF5
            aligned_h5 = sequence_dir / "hdf5" / f"events_cam{cam_idx}.h5"
            if not aligned_h5.exists():
                result["error"] = f"cam{cam_idx}: aligned HDF5 not found: {aligned_h5}"
                return result
            with h5py.File(aligned_h5, 'r') as f:
                ev_t_start_us = f['events/t_ev_start_us'][:]  # (N,) int64
                ev_t_end_us   = f['events/t_ev_end_us'][:]    # (N,) int64

            # Read triggers from RAW first, then use the HDF5 copy as fallback.
            hw_trig_us = None
            try:
                from metavision_core.event_io import RawReader as _RR
                _rr = _RR(str(raw_path))
                while not _rr.is_done():
                    _rr.load_delta_t(100_000)
                _trig = _rr.get_ext_trigger_events()
                if _trig is not None and len(_trig) > 0:
                    _rising = _trig[_trig["p"] == 1]
                    if len(_rising) > 0:
                        hw_trig_us = _rising["t"].astype(np.int64)
                        print(f"  [cam{cam_idx}] HW triggers: {len(hw_trig_us)} rising edges "
                              f"from raw file (need {n_frames})")
            except Exception as _e:
                print(f"  [cam{cam_idx}] WARNING: could not read triggers from raw file: {_e}")

            if hw_trig_us is None:
                with h5py.File(aligned_h5, 'r') as f:
                    if 'events/hw_trigger_times_us' in f:
                        hw_trig_us = f['events/hw_trigger_times_us'][:].astype(np.int64)
                        print(f"  [cam{cam_idx}] HW triggers: {len(hw_trig_us)} from HDF5 fallback")

            if hw_trig_us is None or len(hw_trig_us) < n_frames:
                result["error"] = (
                    f"cam{cam_idx}: need {n_frames} rising-edge hardware triggers, "
                    f"found {0 if hw_trig_us is None else len(hw_trig_us)}"
                )
                return result

            if len(ev_t_start_us) != n_frames:
                print(f"  [cam{cam_idx}] WARNING: HDF5 has {len(ev_t_start_us)} frames "
                      f"but depth has {n_frames}")

            (sequence_dir / "events").mkdir(parents=True, exist_ok=True)

            # Load all raw events for this camera into memory
            events = load_events_from_raw(raw_path)
            raw_t = events['t'].astype(np.int64)

            out_H, out_W = output_hw
            dtype = np.float16 if as_float16 else np.float32

            ch, cw = crop_hw
            cy0 = (EV_H - ch) // 2
            cx0 = (EV_W - cw) // 2

            with h5py.File(voxels_h5, 'w') as vf:
                ds = vf.create_dataset(
                    "voxels",
                    shape=(n_frames, num_bins, out_H, out_W),
                    dtype=dtype,
                    chunks=(1, num_bins, out_H, out_W),
                )
                # Spatial metadata records the native crop and final resize.
                ds.attrs["native_h"] = EV_H
                ds.attrs["native_w"] = EV_W
                ds.attrs["resize_h"] = output_hw[0]
                ds.attrs["resize_w"] = output_hw[1]
                ds.attrs["crop_h"] = crop_hw[0]
                ds.attrs["crop_w"] = crop_hw[1]
                ds.attrs["intrinsics_transform"] = "center_crop_resize"
                ds.attrs["normalized"] = bool(normalize)
                vf.create_dataset("hw_trigger_times_us", data=hw_trig_us[:n_frames])
                trig = hw_trig_us[:n_frames]
                periods = np.diff(trig).astype(np.int64)
                mean_period = int(np.mean(periods)) if len(periods) > 0 else int(1e6 // 30)
                half = mean_period // 2
                win_starts = trig - half
                win_ends   = trig + half

                for frame_idx in tqdm(range(n_frames), desc=f"  cam{cam_idx}", leave=False, position=1, disable=not show_progress):
                    t_start = int(win_starts[frame_idx])
                    t_end   = int(win_ends[frame_idx])

                    i0 = np.searchsorted(raw_t, t_start, side='left')
                    i1 = np.searchsorted(raw_t, t_end,   side='right')

                    voxel = events_to_voxel_grid(
                        events[i0:i1], EV_H, EV_W, num_bins,
                        t_start_us=t_start, t_end_us=t_end,
                        normalize=normalize,
                    )
                    voxel = voxel[:, cy0:cy0+ch, cx0:cx0+cw]
                    voxel = resize_voxel(voxel, output_hw, as_float16=False)
                    if as_float16:
                        voxel = voxel.astype(np.float16)
                    ds[frame_idx] = voxel

        result["success"] = True
        result["n_frames"] = n_frames

    except Exception as e:
        result["error"] = str(e)

    return result


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find valid sequence directories at or recursively below ``data_root``."""
    data_root = Path(data_root)

    def is_sequence(path: Path) -> bool:
        raw_dir = path / "raw_event_data"
        return (
            (path / "hdf5" / "realsense.h5").is_file()
            and raw_dir.is_dir()
            and any(raw_dir.glob("events_cam*.raw"))
        )

    if is_sequence(data_root):
        return [data_root]
    if not data_root.is_dir():
        return []
    return sorted(
        hdf5_dir.parent
        for hdf5_dir in data_root.rglob("hdf5")
        if is_sequence(hdf5_dir.parent)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Precompute voxel grids from raw events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", nargs="+", type=str, default=None,
                       help="Sequence directory(ies) to process")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                       help="Root data directory (will process all subdirs)")
    parser.add_argument("--num_bins", type=int, default=5,
                       help="Number of temporal bins for voxel grid")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing voxel files")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: min(4, n_sequences)). "
                            "Each worker loads a full raw event file into RAM; keep this "
                            "small to avoid OOM. Use 1 to force sequential processing.")
    parser.add_argument("--output_h", type=int, default=PREPROCESS_RESIZE_HW[0],
                       help="Final height after the native center crop")
    parser.add_argument("--output_w", type=int, default=PREPROCESS_RESIZE_HW[1],
                       help="Final width after the native center crop")
    parser.add_argument("--crop_h", type=int, default=PREPROCESS_CROP_HW[0],
                       help="Native center-crop height")
    parser.add_argument("--crop_w", type=int, default=PREPROCESS_CROP_HW[1],
                       help="Native center-crop width")
    parser.add_argument("--float16", action="store_true",
                       help="Store voxels as float16 instead of float32 (2x extra space saving).")
    parser.add_argument("--no_normalize", "--no-normalize", action="store_true",
                       help="Store raw accumulated voxel event counts without mean/std normalization.")
    args = parser.parse_args()

    def _resolve_data_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (_HERE.parent / path).resolve()

    output_hw = (args.output_h, args.output_w)
    crop_hw = (args.crop_h, args.crop_w)
    if min(*output_hw, *crop_hw) <= 0:
        parser.error("crop and output dimensions must be positive")

    # Resolve the native event-camera size before any crop validation. For the
    # current pipeline this is always 1280x720 (W x H); if a sequence-specific HDF5 is
    # already present, prefer that metadata to remain consistent with the data.
    EV_H = 720
    EV_W = 1280

    if args.data_dir:
        candidate_dirs = [_resolve_data_path(d) for d in args.data_dir]
    else:
        candidate_dirs = [_resolve_data_path(args.data_root)]

    for candidate in candidate_dirs:
        if not candidate.exists():
            continue
        direct_event_file = candidate / "hdf5" / "events_cam0.h5"
        event_files = (
            [direct_event_file]
            if direct_event_file.is_file()
            else sorted(candidate.rglob("hdf5/events_cam0.h5"))
        )
        if event_files:
            with h5py.File(event_files[0], "r") as f:
                EV_H = int(f["events"].attrs["height"])
                EV_W = int(f["events"].attrs["width"])
            break

    if crop_hw[0] > EV_H or crop_hw[1] > EV_W:
        parser.error(f"crop {crop_hw} exceeds native event size {(EV_H, EV_W)}")

    # Find sequences
    if args.data_dir:
        sequence_dirs = sorted({
            sequence
            for data_dir in args.data_dir
            for sequence in find_sequence_dirs(_resolve_data_path(data_dir))
        })
    else:
        data_root = _resolve_data_path(args.data_root)
        print(f"Data root: {data_root}")
        sequence_dirs = find_sequence_dirs(data_root)

    if not sequence_dirs:
        if args.data_dir:
            print("No valid sequences found in the provided --data_dir paths.")
        else:
            print(f"No valid sequences found under {data_root}.")
        return

    print(f"Found {len(sequence_dirs)} sequences to process")
    print(f"Voxel bins: {args.num_bins}")
    native_px = EV_W * EV_H
    out_px = output_hw[1] * output_hw[0]
    print(f"Crop → resize  : {crop_hw[1]}×{crop_hw[0]} → {output_hw[1]}×{output_hw[0]} "
          f"({out_px/native_px*100:.1f}% of native {EV_W}×{EV_H}, "
          f"~{native_px/out_px:.1f}x smaller per voxel)")
    if args.float16:
        print("Dtype: float16 (2x additional saving vs float32)")
    normalize = not args.no_normalize
    print(f"Voxel normalization: {'ON' if normalize else 'OFF'}")
    print(f"Overwrite: {args.overwrite}")
    print("HW trigger: required — voxel windows centred on trigger timestamps")
    print()

    # Process sequences
    # Cap workers: each worker loads a full raw event file into RAM, so running
    # too many in parallel causes OOM.  Default to min(4, n_sequences).
    n_workers = args.workers if args.workers is not None else min(4, len(sequence_dirs))

    def _run_sequential(dirs, show_prog=True):
        results = []
        for seq_dir in tqdm(dirs, desc="Processing", position=0, leave=True):
            result = process_sequence(
                seq_dir, args.num_bins, args.overwrite, output_hw, args.float16, crop_hw,
                show_progress=show_prog, normalize=normalize,
            )
            results.append(result)
            if result["success"]:
                tqdm.write(f"  ✓ {result['name']}: {result['n_frames']} frames")
            else:
                tqdm.write(f"  ✗ {result['name']}: {result['error']}")
        return results

    if n_workers == 1 or len(sequence_dirs) == 1:
        results = _run_sequential(sequence_dirs, show_prog=True)
    else:
        # Parallel processing with automatic fallback to sequential on crash
        print(f"Using {n_workers} workers")
        results = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        process_sequence, seq_dir, args.num_bins, args.overwrite,
                        output_hw, args.float16, crop_hw, False, normalize,
                    ): seq_dir
                    for seq_dir in sequence_dirs
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    result = future.result()
                    results.append(result)
                    if result["success"]:
                        tqdm.write(f"  ✓ {result['name']}: {result['n_frames']} frames")
                    else:
                        tqdm.write(f"  ✗ {result['name']}: {result['error']}")
        except BrokenProcessPool:
            completed = {r['name'] for r in results if r['success']}
            remaining = [d for d in sequence_dirs if d.name not in completed]
            print(f"\nWorker process crashed (likely OOM with {n_workers} workers loading "
                  f"raw event files simultaneously).")
            print(f"Falling back to sequential processing for {len(remaining)} remaining "
                  f"sequence(s) …")
            results += _run_sequential(remaining, show_prog=True)
    
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
        sys.exit(1)


if __name__ == "__main__":
    main()
