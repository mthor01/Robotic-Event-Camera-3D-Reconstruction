"""
Main training script for event-to-depth models.

Model architecture is selected via --model (see training/models/ and MODEL_REGISTRY).
Currently implemented: e2depth (Hidalgo-Carrió et al., 3DV 2020).

Expected data layout:
    data/real/<object_name>/
        hdf5/
            realsense.h5              # raw depth (N, H, W) uint16 mm + timestamps
            depth_in_event_frame.h5   # projected depth (N, H, W) float32 metres  [preferred]
            poses.h5                  # EE poses (N, 4, 4) T_base_from_ee          [--use_pose_warp]
        events/
            voxels_cam0.h5            # precomputed event voxels (N, C, H, W)      [precompute_voxels.py]

Prerequisites:
    python3 precompute_voxels.py
    python3 precompute_spatial_mask.py  # if --spatial_mask

Usage:
    python3 train.py --data_root data/real
    python3 train.py --data_dir data/real/bottle data/real/cube_medium
    python3 train.py --data_root data/real --model e2depth --use_pose_warp

Checkpoints and TensorBoard logs land in training/checkpoints/<model>/ by default.
    tensorboard --logdir training/checkpoints/<model>/runs
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import h5py
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import (
    D_MAX, DEPTH_MIN, NUM_BINS,
    DATA_ROOT as _DATA_ROOT, DEFAULT_OUT_DIR,
    CALIB_DIR as _CALIB_DIR,
    TRAIN_RESIZE_HW, TRAIN_CROP_HW, TRAIN_BATCH_SIZE, TRAIN_SEQ_LEN,
)

# Model registry maps --model names to factory functions (see training/models/__init__.py)
from models import MODEL_REGISTRY, MODEL_ARG_REGISTRY, MODEL_DATA_TYPE

# Depth conversion utilities used in validate / log_images
from models.e2depth import (
    E2DepthNet,
    linear_normalized_to_depth,
    depth_to_linear_normalized,
)

# TF32: free speedup on Ampere+ GPUs with negligible numerical impact
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")





# ================= DEFAULT PATHS =================
DATA_ROOT = _DATA_ROOT
# camera_data/ lives one level above training/ in 3d_reconstruction/
CALIB_DIR = Path(__file__).resolve().parent.parent / _CALIB_DIR
# =================================================


# -----------------------------
# Dataset Configuration
# -----------------------------
@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    seq_len: int = 1  # Sequence length for recurrent training
    crop_size: Optional[Tuple[int, int]] = None  # (H, W) center crop applied after resize
    resize_hw: Optional[Tuple[int, int]] = None  # (H, W) resize before crop/augmentation
    depth_max: float = D_MAX  # Maximum depth in meters
    depth_min: float = DEPTH_MIN   # Minimum depth in meters (5cm for tabletop)
    augment: bool = False  # Apply data augmentation during training
    num_bins: int = NUM_BINS  # Number of temporal bins for voxel grid
    use_pose_warp: bool = False  # Warp ConvLSTM hidden states using relative camera pose (from poses.h5)
    spatial_mask: bool = True  # Mask pixels outside cube around EE (from spatial_mask.h5)
    frame_stride: int = 1  # Step size between frames in a sequence (1 = consecutive, k = skip k-1 frames)


class RealDataset(Dataset):
    """
    Dataset for real data recorded with franka_pipeline + synchronised_recording.

    Requires precomputed voxels (run precompute_voxels.py first).
    When ``cfg.use_pose_warp`` is ``True``, loads poses from hdf5/poses.h5
    and computes relative camera transforms per timestep.  These are returned
    alongside the event voxels and used by the model to warp ConvLSTM hidden
    states for ego-motion compensation.

    """
    def __init__(
        self,
        sequence_dir: str,
        cfg: DataConfig,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        ):

        super().__init__()
        self.sequence_dir = Path(sequence_dir)
        self.cfg = cfg
        self.split = split
        
        # --- Depth source ---
        # Always uses the pre-projected depth (event-camera frame, float32 metres).
        # Run project_realsense_to_event.py first to generate this file.
        self.depth_h5_path = self.sequence_dir / "hdf5" / "depth_in_event_frame.h5"
        if not self.depth_h5_path.exists():
            raise FileNotFoundError(
                f"depth_in_event_frame.h5 not found at {self.depth_h5_path}. "
                "Run project_realsense_to_event.py first."
            )

        # Timestamps always come from realsense.h5 regardless of depth source.
        # Recordings use t_global_ms (milliseconds).
        # They are converted to microseconds for a uniform internal representation.
        self._ts_h5_path = self.sequence_dir / "hdf5" / "realsense.h5"
        with h5py.File(self._ts_h5_path, "r") as _tsf:
            self._ts_key = "t_global_ms"
            self._ts_scale = 1000  # ms → µs

        # --- Event voxels (precomputed by precompute_voxels.py) ---
        # Expected: events/voxels_cam0.h5  (dataset "voxels", shape N×C×H×W, float16 or float32)
        self.use_pose_warp = cfg.use_pose_warp
        self.voxels_h5_path: Path = self.sequence_dir / "events" / "voxels_cam0.h5"
        if not self.voxels_h5_path.exists():
            raise FileNotFoundError(
                f"voxels_cam0.h5 not found at {self.voxels_h5_path}. "
                "Run precompute_voxels.py first."
            )

        # HDF5 file handle is opened lazily per DataLoader worker to avoid
        # pickling issues and repeated open/close overhead per sample.
        self._voxels_ds = None

        # --- Camera poses for pose-warp (optional) ---
        # When enabled, each sample returns a per-step relative pose T_curr_from_prev
        # that the models can use to warp states between frames,
        # compensating for camera ego-motion.
        #
        # The recorded poses are T_base_from_ee (robot base → end-effector).
        # We convert them to T_event_from_world (event-camera frame) via:
        #   T_event_from_world[i] = T_event_from_ee  @  inv(T_base_from_ee[i])
        # where T_event_from_ee = T_event_from_rgb @ T_rgb_from_ee  (from calibration).
        self._T_cam_from_world: Optional[np.ndarray] = None  # (N, 4, 4)
        if self.use_pose_warp:
            poses_path = self.sequence_dir / "hdf5" / "poses.h5"
            if not poses_path.exists():
                raise FileNotFoundError(
                    f"--use_pose_warp requires poses.h5. "
                    f"Expected at: {poses_path}"
                )
            T_rgb_from_ee    = np.load(CALIB_DIR / "T_rgb_from_ee.npz")["T"]
            T_event_from_rgb = np.load(CALIB_DIR / "T_event_from_rgb.npz")["T"]
            T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee  # (4, 4) fixed extrinsic chain
            with h5py.File(poses_path, 'r') as pf:
                ee_T = pf["ee_T"][:]  # (N, 4, 4) T_base_from_ee
            T_ee_inv = np.linalg.inv(ee_T)  # (N, 4, 4)  inv(T_base_from_ee)
            # Broadcast: T_event_from_world[i] = T_event_from_ee @ inv(ee_T[i])
            self._T_cam_from_world = np.einsum(
                'ij,njk->nik', T_event_from_ee, T_ee_inv
            ).astype(np.float32)
        
        # Verify files exist
        if not self.depth_h5_path.exists():
            raise FileNotFoundError(f"Depth HDF5 not found: {self.depth_h5_path}")
        
        with h5py.File(self.voxels_h5_path, 'r') as _f:
            n_voxels = int(_f["voxels"].shape[0])
        if n_voxels == 0:
            raise FileNotFoundError(f"Empty voxels dataset in {self.voxels_h5_path}")
        
        # --- Metadata ---
        with h5py.File(self.depth_h5_path, 'r') as f:
            self.n_frames = f["depth"].shape[0]
            self.H = f["depth"].shape[1]
            self.W = f["depth"].shape[2]
        with h5py.File(self._ts_h5_path, 'r') as f:
            self.depth_timestamps = f[self._ts_key][:] // self._ts_scale
        
        if n_voxels != self.n_frames:
            print(f"Warning: {n_voxels} voxels != {self.n_frames} frames")
            self.n_frames = min(n_voxels, self.n_frames)
        
        # --- Spatial mask (optional, precomputed) ---
        self.spatial_mask = cfg.spatial_mask
        self._spatial_h5_path = None
        if self.spatial_mask:
            sp = self.sequence_dir / "hdf5" / "spatial_mask.h5"
            if not sp.exists():
                raise FileNotFoundError(
                    f"--spatial_mask requires precomputed masks. "
                    f"Run: python precompute_spatial_mask.py --data_dir {self.sequence_dir}"
                )
            self._spatial_h5_path = sp
        
        # --- Train / val split ---
        self._compute_valid_indices(val_ratio, seed)
        
        pose_str = "pose_warp" if self.use_pose_warp else "no pose"
        spatial_str = ", spatial_mask" if self.spatial_mask else ""
        print(f"[{self.sequence_dir.name}] {split}: {len(self.indices)} samples "
              f"(frames: {self.n_frames}, depth: projected [linear], {pose_str}{spatial_str}, res: {self.W}x{self.H})")
    
    # The following method computes the train/val split
    # When multiple object sequences are used as input, the split is set to "train" for all of them,
    # and the splitting is done at the dataset level by concatenating them and applying a global split. 
    def _compute_valid_indices(self, val_ratio: float, seed: int):
        """Compute valid starting indices for sequences."""
        seq_len = self.cfg.seq_len
        stride = self.cfg.frame_stride

        # A sequence starting at idx spans frames: idx, idx+stride, ..., idx+(seq_len-1)*stride
        # So the last required frame index is idx + (seq_len - 1) * stride
        idx_max = self.n_frames - 1 - (seq_len - 1) * stride
        if idx_max < 0:
            raise RuntimeError(
                f"Not enough frames ({self.n_frames}) for seq_len={seq_len}, frame_stride={stride}"
            )

        all_indices = np.arange(0, idx_max + 1, dtype=np.int64)
        n_total = len(all_indices)
        
        # Block-based split: group consecutive frames into blocks, then assign
        # whole blocks to train/val.  This prevents temporally adjacent frames
        # from leaking across the split boundary (which a random per-frame split
        # would cause, since adjacent frames are nearly identical).
        rng = np.random.default_rng(seed)
        block_size = max(10, seq_len * stride * 2)
        n_blocks = (n_total + block_size - 1) // block_size
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)
        
        n_val = int(round(n_total * val_ratio))
        val_mask = np.zeros(n_total, dtype=bool)
        val_count = 0
        
        for b in block_ids:
            if val_count >= n_val:
                break
            start = b * block_size
            end = min((b + 1) * block_size, n_total)
            val_mask[start:end] = True
            val_count += end - start
        
        if self.split == "train":
            self.indices = all_indices[~val_mask]
        else:
            self.indices = all_indices[val_mask]
    
    def _get_voxel(self, frame_idx: int) -> np.ndarray:
        """Load one voxel grid from voxels_cam0.h5 as float32 (C, H, W).

        The HDF5 dataset is opened once per worker process and kept open
        for the lifetime of that worker (lazy initialisation via _voxels_ds).
        """
        if self._voxels_ds is None:
            self._voxels_ds = h5py.File(self.voxels_h5_path, 'r')["voxels"]
        v = self._voxels_ds[frame_idx]
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        return v
    
    def _get_depth(self, frame_idx: int) -> np.ndarray:
        """Load one depth frame from depth_in_event_frame.h5 as float32 metres."""
        with h5py.File(self.depth_h5_path, 'r') as f:
            depth = f["depth"][frame_idx].astype(np.float32)
        return depth
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i: int):
        """Return one training sample: a contiguous sequence of seq_len frames.

        Returns a tuple of four tensors, each with a leading time dimension T:
            events : (T, C, H, W)  event voxel grid (C = num_bins)
            depths : (T, 1, H, W)  linearly normalised ground-truth depth in [0, 1]
            masks  : (T, 1, H, W)  binary validity mask (1 = valid pixel)
            poses  : (T, 4, 4)     T_curr_from_prev relative pose (identity at t=0)
        """
        start_idx = int(self.indices[i])
        seq_len = self.cfg.seq_len
        stride = self.cfg.frame_stride

        events_seq = []
        depth_seq = []
        mask_seq = []
        poses_seq = []

        for t in range(seq_len):
            idx = start_idx + t * stride

            voxel = self._get_voxel(idx)   # (C, H, W) float32
            depth = self._get_depth(idx)   # (H, W) float32, metres

            # Validity mask: pixels within the configured depth range
            mask = ((depth > self.cfg.depth_min) & (depth < self.cfg.depth_max)).astype(np.float32)

            # Optionally apply the precomputed spatial mask (a bounding cube
            # around the end-effector workspace, computed by precompute_spatial_mask.py).
            if self.spatial_mask:
                with h5py.File(self._spatial_h5_path, 'r') as sf:
                    sp = sf["mask"][idx]  # (H, W) uint8, 0 = outside workspace
                mask[sp == 0] = 0.0

            # Normalise depth linearly to [0, 1] for network supervision.
            # The inverse transform (linear_normalized_to_depth) is applied
            # inside the loss and metric functions when metric values are needed.
            depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
            depth = (depth - self.cfg.depth_min) / (self.cfg.depth_max - self.cfg.depth_min)
            depth = np.clip(depth, 0, 1)

            # Relative pose T_curr_from_prev used by the model's pose-warp module.
            # At t=0 (first frame of sequence) there is no previous frame, so we
            # pass the identity — the model treats this as "no motion".
            if self.use_pose_warp and self._T_cam_from_world is not None and t > 0:
                T_curr = self._T_cam_from_world[idx]
                T_prev = self._T_cam_from_world[idx - stride]
                T_rel = (T_curr @ np.linalg.inv(T_prev)).astype(np.float32)
            else:
                T_rel = np.eye(4, dtype=np.float32)

            events_seq.append(voxel)
            depth_seq.append(depth[None])  # (1, H, W)
            mask_seq.append(mask[None])
            poses_seq.append(T_rel)

        # Stack sequences: (T, C, H, W)
        events = np.stack(events_seq, axis=0)
        depths = np.stack(depth_seq, axis=0)
        masks = np.stack(mask_seq, axis=0)
        poses = np.stack(poses_seq, axis=0)  # (T, 4, 4)

        # --- Resize ---
        # Depths, masks, and events may already have been precomputed at the
        # final training resolution — if so we skip resizing them.
        if self.cfg.resize_hw is not None:
            rh, rw = self.cfg.resize_hw
            out_H = self.cfg.crop_size[0] if self.cfg.crop_size is not None else rh
            out_W = self.cfg.crop_size[1] if self.cfg.crop_size is not None else rw
            dep_H, dep_W = depths.shape[2], depths.shape[3]
            # Only resize depth/mask if not already at the final target resolution
            if dep_H != out_H or dep_W != out_W:
                depths = F.interpolate(torch.from_numpy(depths), size=(rh, rw), mode="bilinear", align_corners=False).numpy()
                masks  = F.interpolate(torch.from_numpy(masks),  size=(rh, rw), mode="nearest").numpy()
            ev_H, ev_W = events.shape[2], events.shape[3]
            # Only resize events if they are not already at the target resolution
            if (ev_H != rh or ev_W != rw) and (ev_H != out_H or ev_W != out_W):
                events = F.interpolate(torch.from_numpy(events), size=(rh, rw), mode="bilinear", align_corners=False).numpy()

        # Use depths as the reference for current spatial size — it is always
        # resized, unlike events which may already be at crop resolution.
        cur_H = depths.shape[2]
        cur_W = depths.shape[3]

        # Center crop (applied to both train and val whenever crop_size is set)
        if self.cfg.crop_size is not None:
            ch, cw = self.cfg.crop_size
            if cur_H < ch or cur_W < cw:
                raise ValueError(
                    f"Center crop {cw}x{ch} larger than image {cur_W}x{cur_H}"
                )
            y0 = (cur_H - ch) // 2
            x0 = (cur_W - cw) // 2
            depths = depths[:, :, y0:y0+ch, x0:x0+cw]
            masks  = masks[:, :, y0:y0+ch, x0:x0+cw]
            # Events may already be at crop size (precomputed at training resolution).
            if events.shape[2] != ch or events.shape[3] != cw:
                events = events[:, :, y0:y0+ch, x0:x0+cw]

        return (
            torch.from_numpy(events).float(),
            torch.from_numpy(depths).float(),
            torch.from_numpy(masks).float(),
            torch.from_numpy(poses).float(),
        )


def create_multi_sequence_dataset(
    sequence_dirs: List[str],
    cfg: DataConfig,
    split: str = "train",
    val_ratio: float = 0.2,
) -> Dataset:
    """Create concatenated dataset from multiple sequences."""
    datasets = []
    for seq_dir in sequence_dirs:
        try:
            ds = RealDataset(seq_dir, cfg, split=split, val_ratio=val_ratio)
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Skipping {seq_dir}: {e}")
    
    if not datasets:
        raise RuntimeError("No valid datasets found!")
    
    return ConcatDataset(datasets)


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """
    Find valid real data sequence directories under data_root.

    Required layout:
        hdf5/depth_in_event_frame.h5   # projected depth (run project_realsense_to_event.py)
        events/voxels_cam0.h5          # precomputed voxels (run precompute_voxels.py)
    """
    sequence_dirs = []
    for d in data_root.iterdir():
        if not d.is_dir():
            continue
        has_depth = (d / "hdf5" / "depth_in_event_frame.h5").exists()
        has_voxels = (d / "events" / "voxels_cam0.h5").exists()
        if has_depth and has_voxels:
            sequence_dirs.append(d)
    return sorted(sequence_dirs)


# -----------------------------
# Debug Visualization
# -----------------------------
def debug_visualize(
    sequence_dir: str,
    n_samples: int = 3,
    out_path: str = "debug_viz.png",
    num_bins: int = 5,
    seed: int = None,
    resize_hw: Optional[Tuple[int, int]] = None,
    crop_size: Optional[Tuple[int, int]] = None,
    use_pose: bool = False,
    spatial_mask: bool = False,
) -> None:
    """
    Save a debug PNG with n_samples × 2 rows (raw + preprocessed per timestamp).

    RAW row columns:
      1. Depth – Realsense raw            (plasma, metres)
      2. Depth Mask                        (gray)
      3. Projected Depth – event plane     (plasma, metres)  [N/A if absent]
      4. Event Plane Mask                  (gray)            [N/A if absent]
      5. Events Accumulated – sum of bins  (gray)
      6. RGB (Realsense raw)               [N/A if absent]
      7. (blank)
      8. Depth + Events overlay            (plasma + red/blue alpha)
      9. Depth × Mask overlay              (plasma, masked)

    PREPROCESSED row columns (after resize then center-crop):
      1. Projected Depth – preprocessed    (plasma)
      2. Event Plane Mask – preprocessed   (gray)
      3. Events Accumulated – preprocessed (gray)
      4. RGB – preprocessed                [N/A if absent]
      5. Pose Depth – raw event resolution (viridis)  [N/A if --use_pose off]
      6. Pose Depth – preprocessed         (viridis)  [N/A if --use_pose off]
      7. Total Mask – preprocessed         (gray)     [N/A if no depth projected]
      8. Depth + Events overlay – preprocessed
      9. Depth × Mask overlay – preprocessed

    Resolution (W×H) is annotated in the top-left corner of every panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_dir = Path(sequence_dir)
    rng = np.random.default_rng(seed)

    # ---- Locate raw depth & timestamps ----
    realsense_path = seq_dir / "hdf5" / "realsense.h5"
    projected_path = seq_dir / "hdf5" / "depth_in_event_frame.h5"

    if not realsense_path.exists():
        raise FileNotFoundError(f"No realsense.h5 found in {seq_dir / 'hdf5'}")
    raw_depth_path = realsense_path
    raw_depth_key = "depth"
    with h5py.File(realsense_path, "r") as _tsf:
        if "t_global_ms" in _tsf:
            ts_key = "t_global_ms"
            ts_scale = 1000  # ms → µs
        else:
            ts_key = "t_sys_ns"
            ts_scale = 1000  # ns → µs

    # ---- Locate RGB ----
    rgb_path: Optional[Path] = None
    rgb_key: Optional[str] = None
    for _rp, _rk in [
        (seq_dir / "hdf5" / "realsense.h5", "rgb"),
        (seq_dir / "hdf5" / "rgb.h5", "rgb"),
    ]:
        if _rp.exists():
            with h5py.File(_rp, "r") as _f:
                if _rk in _f:
                    rgb_path, rgb_key = _rp, _rk
                    break

    # ---- Locate precomputed voxels ----
    voxels_h5_path: Optional[Path] = None
    voxels_dir: Optional[Path] = None
    _h5 = seq_dir / "events" / "voxels_cam0.h5"
    if _h5.exists():
        voxels_h5_path = _h5
    else:
        for _vd in [seq_dir / "events" / "voxels_cam0", seq_dir / "events" / "voxels"]:
            if _vd.exists() and list(_vd.glob("voxel_*.npy")):
                voxels_dir = _vd
                break
    # ---- Load metadata ----
    with h5py.File(raw_depth_path, "r") as _f:
        n_frames   = _f[raw_depth_key].shape[0]
        depth_H    = int(_f[raw_depth_key].shape[1])
        depth_W    = int(_f[raw_depth_key].shape[2])

    proj_H: Optional[int] = None
    proj_W: Optional[int] = None
    if projected_path.exists():
        with h5py.File(projected_path, "r") as _f:
            proj_H = int(_f["depth"].shape[1])
            proj_W = int(_f["depth"].shape[2])

    # ---- Pose voxels (precomputed, optional) ----
    pose_voxels_dir: Optional[Path] = None
    if use_pose:
        _pvd = seq_dir / "events" / "voxels_pose_cam0"
        if _pvd.exists() and list(_pvd.glob("voxel_*.npy")):
            pose_voxels_dir = _pvd
        else:
            print("[debug_viz] Warning: voxels_pose_cam0 not found, skipping pose depth")

    # ---- Spatial mask (precomputed) ----
    spatial_mask_path: Optional[Path] = seq_dir / "hdf5" / "spatial_mask.h5"
    if not spatial_mask_path.exists():
        spatial_mask_path = None
        if spatial_mask:
            print("[debug_viz] Warning: spatial_mask.h5 not found, skipping spatial mask")

    # ---- Sample random frame indices ----
    n_samples = min(n_samples, n_frames)
    frame_indices: List[int] = sorted(
        rng.choice(n_frames, size=n_samples, replace=False).tolist()
    )

    # ---- Helpers ----
    def _apply_resize_crop(img: np.ndarray) -> np.ndarray:
        """img: (H, W) or (H, W, C) — returns resized+cropped numpy array."""
        is_rgb = img.ndim == 3
        # add batch+channel dims expected by F.interpolate: (1, C, H, W)
        if is_rgb:
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        else:
            t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        if resize_hw is not None:
            mode = "bilinear" if is_rgb or img.dtype != bool else "nearest"
            t = F.interpolate(t, size=resize_hw, mode=mode, align_corners=False if mode == "bilinear" else None)
        if crop_size is not None:
            ch, cw = crop_size
            cur_H2, cur_W2 = t.shape[2], t.shape[3]
            y0 = (cur_H2 - ch) // 2
            x0 = (cur_W2 - cw) // 2
            t = t[:, :, y0:y0 + ch, x0:x0 + cw]
        out = t.squeeze(0).numpy()
        if is_rgb:
            out = np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8)
        else:
            out = out.squeeze(0)
        return out

    # ---- Layout ----
    # Each sample occupies 2 rows: raw (top) + preprocessed (bottom)
    N_COLS = 9
    N_ROWS = n_samples * 2
    row_h = 3.5
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 4.2, N_ROWS * row_h))
    if N_ROWS == 1:
        axes = axes[np.newaxis, :]

    preproc_label = ""
    if resize_hw is not None:
        preproc_label += f"resize→{resize_hw[1]}×{resize_hw[0]}"
    if crop_size is not None:
        if preproc_label:
            preproc_label += " + "
        preproc_label += f"center-crop→{crop_size[1]}×{crop_size[0]}"
    if not preproc_label:
        preproc_label = "no resize/crop"

    def _annotate(ax, W: int, H: int, title: str, frame_idx: Optional[int] = None) -> None:
        full_title = f"[HDF idx {frame_idx}]  {title}" if frame_idx is not None else title
        ax.set_title(full_title, fontsize=8, pad=3)
        # Top-left: resolution
        ax.text(
            0.01, 0.99, f"{W}×{H}",
            color="white", fontsize=7, fontweight="bold",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65),
        )
        # Top-right: label
        ax.text(
            0.99, 0.99, title,
            color="white", fontsize=7, fontweight="bold",
            transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65),
        )
        ax.axis("off")

    def _na(ax, title: str) -> None:
        ax.set_title(title, fontsize=8, pad=3)
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")

    def _blank(ax) -> None:
        ax.axis("off")

    for sample_i, frame_idx in enumerate(frame_indices):
        raw_row = sample_i * 2          # top row  — raw data
        pre_row = sample_i * 2 + 1     # bottom row — after preprocessing

        # ================================================================
        # RAW ROW
        # ================================================================

        # --- 1. Raw depth ---
        with h5py.File(raw_depth_path, "r") as _f:
            raw_d = _f[raw_depth_key][frame_idx].astype(np.float32)
        raw_d_m = raw_d / 1000.0 if raw_d.max() > 100.0 else raw_d
        ax = axes[raw_row, 0]
        ax.imshow(raw_d_m, cmap="plasma", vmin=0.0, vmax=2.5)
        _annotate(ax, depth_W, depth_H, "Depth (Realsense)", frame_idx=frame_idx)

        # --- 2. Raw depth mask ---
        raw_mask_arr = ((raw_d_m > 0.01) & (raw_d_m < 5.0)).astype(np.float32)
        ax = axes[raw_row, 1]
        ax.imshow(raw_mask_arr, cmap="gray", vmin=0, vmax=1)
        _annotate(ax, depth_W, depth_H, "Depth Mask")

        # --- 3. Projected depth (event plane) ---
        proj_d: Optional[np.ndarray] = None
        ax = axes[raw_row, 2]
        if projected_path.exists():
            with h5py.File(projected_path, "r") as _f:
                proj_d = _f["depth"][frame_idx].astype(np.float32)
            ax.imshow(proj_d, cmap="plasma", vmin=0.0, vmax=2.5)
            _annotate(ax, proj_W, proj_H, "Projected Depth")
        else:
            _na(ax, "Projected Depth")

        # --- 4. Event plane mask ---
        ax = axes[raw_row, 3]
        proj_mask_arr: Optional[np.ndarray] = None
        if proj_d is not None:
            proj_mask_arr = ((proj_d > 0.01) & (proj_d < 5.0)).astype(np.float32)
            ax.imshow(proj_mask_arr, cmap="gray", vmin=0, vmax=1)
            _annotate(ax, proj_W, proj_H, "Event Plane Mask")
        else:
            _na(ax, "Event Plane Mask")

        # --- 5. Accumulated events ---
        voxel: Optional[np.ndarray] = None
        ax = axes[raw_row, 4]
        if voxels_h5_path is not None:
            with h5py.File(voxels_h5_path, 'r') as _f:
                voxel = _f["voxels"][frame_idx].astype(np.float32)
        elif voxels_dir is not None:
            vp = voxels_dir / f"voxel_{frame_idx:06d}.npy"
            if vp.exists():
                voxel = np.load(vp)
        if voxel is not None:
            accum = voxel.sum(axis=0)
            ev_H_v, ev_W_v = accum.shape
            vmax_ev = float(max(np.abs(accum).max(), 1e-6))
            ax.imshow(accum, cmap="gray", vmin=-vmax_ev, vmax=vmax_ev)
            _annotate(ax, ev_W_v, ev_H_v, "Events (Accumulated)")
        else:
            _na(ax, "Events (Accumulated)")

        # --- 6. RGB (Realsense raw) ---
        rgb_frame_raw: Optional[np.ndarray] = None
        ax = axes[raw_row, 5]
        if rgb_path is not None:
            with h5py.File(rgb_path, "r") as _f:
                rgb_frame_raw = _f[rgb_key][frame_idx]
            if rgb_frame_raw.dtype != np.uint8:
                rgb_frame_raw = np.clip(rgb_frame_raw, 0, 255).astype(np.uint8)
            rgb_H2, rgb_W2 = int(rgb_frame_raw.shape[0]), int(rgb_frame_raw.shape[1])
            ax.imshow(rgb_frame_raw)
            _annotate(ax, rgb_W2, rgb_H2, "RGB")
        else:
            _na(ax, "RGB")

        # --- 7. (blank) ---
        _blank(axes[raw_row, 6])

        # --- 8. Depth + Events overlay (raw projected depth + raw events) ---
        ax = axes[raw_row, 7]
        if proj_d is not None and voxel is not None:
            ev_accum_ov = voxel.sum(axis=0)  # (H, W)
            # Resize events to projected depth resolution if needed
            if ev_accum_ov.shape != proj_d.shape:
                ev_accum_ov = cv2.resize(ev_accum_ov, (proj_W, proj_H), interpolation=cv2.INTER_LINEAR)
            plasma_cm = plt.get_cmap("plasma")
            depth_rgba_ov = plasma_cm(np.clip(proj_d / 2.5, 0, 1))  # (H, W, 4)
            ev_abs_ov = np.abs(ev_accum_ov)
            ev_scale = ev_abs_ov.max() + 1e-6
            ev_rgba_ov = np.zeros((*ev_accum_ov.shape, 4), dtype=np.float32)
            ev_rgba_ov[..., 0] = np.clip(ev_accum_ov, 0, None) / ev_scale   # red   = positive
            ev_rgba_ov[..., 2] = np.clip(-ev_accum_ov, 0, None) / ev_scale  # blue  = negative
            ev_rgba_ov[..., 3] = np.clip(ev_abs_ov / ev_scale, 0, 1) * 0.7  # alpha = magnitude
            ax.imshow(depth_rgba_ov)
            ax.imshow(ev_rgba_ov)
            _annotate(ax, proj_W, proj_H, "Depth + Events")
        else:
            _na(ax, "Depth + Events")

        # --- 9. Depth × Mask overlay (projected depth zeroed outside mask) ---
        ax = axes[raw_row, 8]
        if proj_d is not None and proj_mask_arr is not None:
            ax.imshow(proj_d * proj_mask_arr, cmap="plasma", vmin=0.0, vmax=2.5)
            _annotate(ax, proj_W, proj_H, "Depth × Mask")
        else:
            _na(ax, "Depth × Mask")

        # Row label on the left
        axes[raw_row, 0].set_ylabel(
            f"sample {sample_i}  |  RAW", fontsize=9, rotation=90, labelpad=6
        )
        axes[raw_row, 0].axis("off")  # restore after set_ylabel touched it
        _annotate(axes[raw_row, 0], depth_W, depth_H, "Depth (Realsense)", frame_idx=frame_idx)

        # ================================================================
        # PREPROCESSED ROW  (resize → center-crop applied to event-space data)
        # ================================================================

        # --- col 0: projected depth after preprocessing ---
        ax = axes[pre_row, 0]
        if proj_d is not None:
            pd_pre = _apply_resize_crop(proj_d)
            ph, pw = pd_pre.shape
            ax.imshow(pd_pre, cmap="plasma", vmin=0.0, vmax=2.5)
            _annotate(ax, pw, ph, f"Proj Depth  [{preproc_label}]")
        else:
            _na(ax, f"Proj Depth  [{preproc_label}]")

        # --- col 1: event plane mask after preprocessing ---
        ax = axes[pre_row, 1]
        if proj_mask_arr is not None:
            pm_pre = _apply_resize_crop(proj_mask_arr)
            ph, pw = pm_pre.shape
            ax.imshow(pm_pre, cmap="gray", vmin=0, vmax=1)
            _annotate(ax, pw, ph, f"Event Mask  [{preproc_label}]")
        else:
            _na(ax, f"Event Mask  [{preproc_label}]")

        # --- col 2: accumulated events after preprocessing ---
        ax = axes[pre_row, 2]
        if voxel is not None:
            accum_pre = _apply_resize_crop(voxel.sum(axis=0))
            eh, ew = accum_pre.shape
            vmax_ev2 = float(max(np.abs(accum_pre).max(), 1e-6))
            ax.imshow(accum_pre, cmap="gray", vmin=-vmax_ev2, vmax=vmax_ev2)
            _annotate(ax, ew, eh, f"Events  [{preproc_label}]")
        else:
            _na(ax, f"Events  [{preproc_label}]")

        # --- col 3: RGB after preprocessing ---
        ax = axes[pre_row, 3]
        if rgb_frame_raw is not None:
            rgb_pre = _apply_resize_crop(rgb_frame_raw)
            rh2, rw2 = int(rgb_pre.shape[0]), int(rgb_pre.shape[1])
            ax.imshow(rgb_pre)
            _annotate(ax, rw2, rh2, f"RGB  [{preproc_label}]")
        else:
            _na(ax, f"RGB  [{preproc_label}]")

        # --- col 4: pose depth (raw, event resolution) ---
        ax = axes[pre_row, 4]
        pose_depth_raw: Optional[np.ndarray] = None
        if pose_voxels_dir is not None:
            pvp = pose_voxels_dir / f"voxel_{frame_idx:06d}.npy"
            if pvp.exists():
                pose_voxel = np.load(pvp)
                pose_depth_raw = pose_voxel[-1]  # last channel is pose depth
                pdh0, pdw0 = pose_depth_raw.shape
                ax.imshow(pose_depth_raw, cmap="viridis", vmin=0, vmax=1)
                _annotate(ax, pdw0, pdh0, "Pose Depth (raw)")
            else:
                _na(ax, "Pose Depth (raw)")
        else:
            _na(ax, "Pose Depth (raw)")

        # --- col 5: pose depth (preprocessed) ---
        ax = axes[pre_row, 5]
        if pose_depth_raw is not None:
            pd_pre2 = _apply_resize_crop(pose_depth_raw)
            pdh, pdw = pd_pre2.shape
            ax.imshow(pd_pre2, cmap="viridis", vmin=0, vmax=1)
            _annotate(ax, pdw, pdh, f"Pose Depth  [{preproc_label}]")
        else:
            _na(ax, f"Pose Depth  [{preproc_label}]")

        axes[pre_row, 0].set_ylabel(
            f"sample {sample_i}  |  PREPROCESSED", fontsize=9, rotation=90, labelpad=6
        )
        axes[pre_row, 0].axis("off")
        if proj_d is not None:
            _annotate(axes[pre_row, 0], pw, ph, f"Proj Depth  [{preproc_label}]")
        else:
            _na(axes[pre_row, 0], f"Proj Depth  [{preproc_label}]")

        # --- col 6: total mask (depth range + optional spatial mask) after preprocessing ---
        # (variables pw, ph, pd_pre, pm_pre, accum_pre are set conditionally above)
        ax = axes[pre_row, 6]
        if proj_mask_arr is not None:
            total_mask = proj_mask_arr.copy()  # depth range mask
            if spatial_mask and spatial_mask_path is not None:
                with h5py.File(spatial_mask_path, 'r') as _sf:
                    sp_raw = _sf["mask"][frame_idx].astype(np.float32)
                total_mask[sp_raw == 0] = 0.0
            tm_pre = _apply_resize_crop(total_mask)
            tmh, tmw = tm_pre.shape
            ax.imshow(tm_pre, cmap="gray", vmin=0, vmax=1)
            parts = ["depth"]
            if spatial_mask and spatial_mask_path is not None:
                parts.append("spatial")
            label = f"Total Mask ({' + '.join(parts)})"
            _annotate(ax, tmw, tmh, f"{label}  [{preproc_label}]")
        else:
            _na(ax, f"Total Mask  [{preproc_label}]")

        # --- col 7: depth + events overlay (preprocessed) ---
        ax = axes[pre_row, 7]
        if proj_d is not None and voxel is not None:
            # pd_pre and accum_pre are defined above when proj_d/voxel are not None
            plasma_cm = plt.get_cmap("plasma")
            depth_rgba_pre_ov = plasma_cm(np.clip(pd_pre / 2.5, 0, 1))  # (H, W, 4)
            ev_abs_pre_ov = np.abs(accum_pre)
            ev_scale_pre = ev_abs_pre_ov.max() + 1e-6
            ev_rgba_pre_ov = np.zeros((*accum_pre.shape, 4), dtype=np.float32)
            ev_rgba_pre_ov[..., 0] = np.clip(accum_pre, 0, None) / ev_scale_pre
            ev_rgba_pre_ov[..., 2] = np.clip(-accum_pre, 0, None) / ev_scale_pre
            ev_rgba_pre_ov[..., 3] = np.clip(ev_abs_pre_ov / ev_scale_pre, 0, 1) * 0.7
            ov_h, ov_w = pd_pre.shape
            ax.imshow(depth_rgba_pre_ov)
            ax.imshow(ev_rgba_pre_ov)
            _annotate(ax, ov_w, ov_h, f"Depth + Events  [{preproc_label}]")
        else:
            _na(ax, f"Depth + Events  [{preproc_label}]")

        # --- col 8: depth × mask overlay (preprocessed) ---
        ax = axes[pre_row, 8]
        if proj_d is not None and proj_mask_arr is not None:
            # pd_pre and pm_pre are defined above
            dm_h, dm_w = pd_pre.shape
            ax.imshow(pd_pre * pm_pre, cmap="plasma", vmin=0.0, vmax=2.5)
            _annotate(ax, dm_w, dm_h, f"Depth × Mask  [{preproc_label}]")
        else:
            _na(ax, f"Depth × Mask  [{preproc_label}]")

    fig.suptitle(
        f"Debug — {seq_dir.name}   "
        f"(HDF frame indices: {', '.join(str(i) for i in frame_indices)})  "
        f"— preprocessing: {preproc_label}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[debug_viz] Saved → {out_path}")


# -----------------------------
# GPU Monitoring
# -----------------------------
def get_gpu_stats(device: torch.device) -> Dict[str, float]:
    """Return GPU utilization (%) and VRAM usage (MB) for the given device."""
    stats: Dict[str, float] = {}
    if device.type != "cuda":
        return stats

    gpu_idx = device.index if device.index is not None else torch.cuda.current_device()

    stats["vram_used_mb"] = torch.cuda.memory_allocated(gpu_idx) / 1024 ** 2
    stats["vram_reserved_mb"] = torch.cuda.memory_reserved(gpu_idx) / 1024 ** 2

    return stats


# ===================================
# MVS-specific utilities and dataset
# ===================================

def _load_mvs_calibration(calib_dir: Path) -> dict:
    """Load event-camera intrinsics and T_event_from_ee for MVS projection."""
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K_event = ev["camera_matrix"].astype(np.float32)
    ev_size = ev["image_size"]                          # [W, H]
    T_rgb_from_ee    = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float32)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float32)
    T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee
    return {
        "K_event": K_event,
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
        "T_event_from_ee": T_event_from_ee,
    }


def _mvs_make_depth_values(depth_min: float, depth_interval: float, num_depth: int) -> np.ndarray:
    return depth_min + np.arange(num_depth, dtype=np.float32) * depth_interval


def _mvs_compute_proj_mat(
    K: np.ndarray,
    T_cam_from_world: np.ndarray,
    orig_hw: Tuple[int, int],
    target_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Compute scaled 3x4 projection matrix K_scaled @ [R | t]."""
    K = K.copy()
    if target_hw is not None:
        orig_h, orig_w = orig_hw
        new_h,  new_w  = target_hw
        K[0] *= new_w / orig_w
        K[1] *= new_h / orig_h
    return (K @ T_cam_from_world[:3, :]).astype(np.float32)


class RealMVSDataset(Dataset):
    """
    MVSNet dataset built from real robot-recording data.

    Multi-view groups: one reference frame + (num_views - 1) source frames
    sampled at ±view_interval steps.  Camera projection matrices are derived
    from robot EE poses and the event-camera calibration.
    """

    def __init__(
        self,
        sequence_dir,
        calib: dict,
        num_views: int = 5,
        num_depth: int = 192,
        depth_min: float = DEPTH_MIN,
        depth_max: float = D_MAX,
        resize_hw: Optional[Tuple[int, int]] = None,
        view_interval: int = 5,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.seq_dir       = Path(sequence_dir)
        self.calib         = calib
        self.num_views     = num_views
        self.num_depth     = num_depth
        self.depth_min     = depth_min
        self.depth_max     = depth_max
        self.resize_hw     = resize_hw
        self.view_interval = view_interval
        self.split         = split
        self.depth_interval = (depth_max - depth_min) / num_depth

        hdf5 = self.seq_dir / "hdf5"
        self._voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self._depth_path  = hdf5 / "depth_in_event_frame.h5"
        self._pose_path   = hdf5 / "poses.h5"
        self._mask_path   = hdf5 / "spatial_mask.h5"

        for p in (self._voxels_path, self._depth_path, self._pose_path):
            if not p.exists():
                raise FileNotFoundError(p)
        self._has_spatial_mask = self._mask_path.exists()

        with h5py.File(self._pose_path, "r") as pf:
            ee_T = pf["ee_T"][:].astype(np.float32)
        T_event_from_ee = calib["T_event_from_ee"]
        T_ee_inv = np.linalg.inv(ee_T)
        self._T_cam = np.einsum("ij,njk->nik", T_event_from_ee, T_ee_inv).astype(np.float32)

        with h5py.File(self._depth_path, "r") as df:
            self.n_frames = df["depth"].shape[0]
            self._orig_h  = df["depth"].shape[1]
            self._orig_w  = df["depth"].shape[2]

        # Source-frame offsets: symmetric around reference
        n_src = num_views - 1
        half  = n_src // 2
        offsets: List[int] = []
        for k in range(1, half + 1):
            offsets += [k * view_interval, -k * view_interval]
        if n_src % 2 == 1:
            offsets.append((half + 1) * view_interval)
        self._src_offsets = offsets[:n_src]

        max_abs_offset = max(abs(o) for o in self._src_offsets) if self._src_offsets else 0
        valid = np.arange(max_abs_offset, self.n_frames - max_abs_offset, dtype=np.int64)

        rng = np.random.default_rng(seed)
        block_size = max(10, max_abs_offset * 2 + 1)
        n_blocks   = max(1, (len(valid) + block_size - 1) // block_size)
        block_ids  = np.arange(n_blocks)
        rng.shuffle(block_ids)
        n_val    = max(1, int(round(len(valid) * val_ratio)))
        val_mask = np.zeros(len(valid), dtype=bool)
        val_count = 0
        for b in block_ids:
            if val_count >= n_val:
                break
            s = b * block_size
            e = min((b + 1) * block_size, len(valid))
            val_mask[s:e] = True
            val_count += e - s

        self.indices = valid[~val_mask] if split == "train" else valid[val_mask]
        print(
            f"[{self.seq_dir.name}] MVS {split}: {len(self.indices)} samples "
            f"(frames: {self.n_frames}, views: {num_views}, interval: {view_interval})"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_voxel(self, idx: int) -> np.ndarray:
        with h5py.File(self._voxels_path, "r") as f:
            vox = f["voxels"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            vox = F.interpolate(
                torch.from_numpy(vox).unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
        return vox

    def _load_depth(self, idx: int) -> np.ndarray:
        with h5py.File(self._depth_path, "r") as f:
            depth = f["depth"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            depth = F.interpolate(
                torch.from_numpy(depth).unsqueeze(0).unsqueeze(0), size=(rh, rw), mode="nearest"
            ).squeeze().numpy()
        return depth

    def _load_spatial_mask(self, idx: int) -> Optional[np.ndarray]:
        if not self._has_spatial_mask:
            return None
        with h5py.File(self._mask_path, "r") as f:
            sp = f["mask"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            sp = F.interpolate(
                torch.from_numpy(sp).unsqueeze(0).unsqueeze(0), size=(rh, rw), mode="nearest"
            ).squeeze().numpy()
        return sp

    def __getitem__(self, i: int):
        ref_idx  = int(self.indices[i])
        view_ids = [ref_idx] + [ref_idx + o for o in self._src_offsets]
        orig_hw  = (self._orig_h, self._orig_w)

        imgs: List[np.ndarray]      = []
        proj_mats: List[np.ndarray] = []
        for vid in view_ids:
            imgs.append(self._load_voxel(vid))
            proj_mats.append(_mvs_compute_proj_mat(
                self.calib["K_event"], self._T_cam[vid], orig_hw, self.resize_hw
            ))

        depth_values = _mvs_make_depth_values(self.depth_min, self.depth_interval, self.num_depth)
        depth = self._load_depth(ref_idx)
        mask  = ((depth > self.depth_min) & (depth < self.depth_max)).astype(np.float32)
        sp    = self._load_spatial_mask(ref_idx)
        if sp is not None:
            mask[sp == 0] = 0.0

        return {
            "imgs":         torch.from_numpy(np.stack(imgs).astype(np.float32)),
            "proj_mats":    torch.from_numpy(np.stack(proj_mats).astype(np.float32)),
            "depth_values": torch.from_numpy(depth_values),
            "depth":        torch.from_numpy(depth),
            "mask":         torch.from_numpy(mask),
        }


def build_mvs_datasets(sequence_dirs: List[Path], args) -> Tuple[Dataset, Dataset]:
    """Build train/val RealMVSDataset from a list of sequence directories."""
    calib = _load_mvs_calibration(CALIB_DIR)
    resize_hw = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize_hw = (args.resize_h, args.resize_w)

    train_datasets: List[Dataset] = []
    val_datasets:   List[Dataset] = []
    for sd in sequence_dirs:
        try:
            train_datasets.append(RealMVSDataset(
                sd, calib, num_views=args.num_views, num_depth=args.num_depth,
                depth_min=args.depth_min, depth_max=args.depth_max,
                resize_hw=resize_hw, view_interval=args.view_interval,
                split="train", val_ratio=args.val_ratio,
            ))
            val_datasets.append(RealMVSDataset(
                sd, calib, num_views=args.num_views, num_depth=args.num_depth,
                depth_min=args.depth_min, depth_max=args.depth_max,
                resize_hw=resize_hw, view_interval=args.view_interval,
                split="val", val_ratio=args.val_ratio,
            ))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Warning: skipping {sd} for MVS: {exc}")
    if not train_datasets:
        raise RuntimeError("No valid sequences for MVS training.")
    train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val   = ConcatDataset(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]
    return train, val


# ===================================
# Pose-supervised dataset
# ===================================

class RealPoseDataset(Dataset):
    """Dataset for PoseUNet: one reference frame + N-1 source frames.

    Each sample:
        tgt_voxels  (C, H, W)         target (reference) event voxels
        src_voxels  (N_src, C, H, W)  source event voxels
        depth_norm  (1, H, W)         GT depth normalised to [0, 1]
        mask        (1, H, W)         validity mask
        T_s_from_t  (N_src, 4, 4)     T_source_from_target per source frame

    Frame layout for n_frames=5, frame_offset=k:
        offsets: [-2k, -k, 0 (ref), +k, +2k]
        source offsets: [-2k, -k, +k, +2k]  (all except 0)
    """

    def __init__(
        self,
        sequence_dir,
        n_frames:     int   = 5,
        frame_offset: int   = 15,
        depth_min:    float = DEPTH_MIN,
        depth_max:    float = D_MAX,
        resize_hw:    Optional[Tuple[int, int]] = None,
        split:        str   = "train",
        val_ratio:    float = 0.1,
        seed:         int   = 42,
    ):
        self.seq_dir      = Path(sequence_dir)
        self.n_frames     = n_frames
        self.frame_offset = frame_offset
        self.depth_min    = depth_min
        self.depth_max    = depth_max
        self.resize_hw    = resize_hw

        half = n_frames // 2
        # All offsets including 0 (reference); source offsets exclude 0
        all_offsets  = [i * frame_offset for i in range(-half, -half + n_frames)]
        self.src_offsets: List[int] = [o for o in all_offsets if o != 0]

        hdf5 = self.seq_dir / "hdf5"
        self._voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self._depth_path  = hdf5 / "depth_in_event_frame.h5"
        self._pose_path   = hdf5 / "poses.h5"
        self._mask_path   = hdf5 / "spatial_mask.h5"

        for p in (self._voxels_path, self._depth_path, self._pose_path):
            if not p.exists():
                raise FileNotFoundError(p)
        self._has_spatial_mask = self._mask_path.exists()

        # Absolute camera poses: T_cam_from_world[i]
        T_rgb_from_ee    = np.load(CALIB_DIR / "T_rgb_from_ee.npz")["T"].astype(np.float32)
        T_event_from_rgb = np.load(CALIB_DIR / "T_event_from_rgb.npz")["T"].astype(np.float32)
        T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee
        with h5py.File(self._pose_path, "r") as pf:
            ee_T = pf["ee_T"][:].astype(np.float32)
        T_ee_inv = np.linalg.inv(ee_T)
        self._T_cam = np.einsum("ij,njk->nik", T_event_from_ee, T_ee_inv).astype(np.float32)

        with h5py.File(self._depth_path, "r") as df:
            self.total_frames = df["depth"].shape[0]
            self._orig_h      = df["depth"].shape[1]
            self._orig_w      = df["depth"].shape[2]

        max_abs_offset = max(abs(o) for o in self.src_offsets) if self.src_offsets else 0
        valid = np.arange(max_abs_offset, self.total_frames - max_abs_offset, dtype=np.int64)

        rng        = np.random.default_rng(seed)
        block_size = max(10, max_abs_offset * 2 + 1)
        n_blocks   = max(1, (len(valid) + block_size - 1) // block_size)
        block_ids  = np.arange(n_blocks)
        rng.shuffle(block_ids)
        n_val      = max(1, int(round(len(valid) * val_ratio)))
        val_mask   = np.zeros(len(valid), dtype=bool)
        val_count  = 0
        for b in block_ids:
            if val_count >= n_val:
                break
            s = b * block_size
            e = min((b + 1) * block_size, len(valid))
            val_mask[s:e] = True
            val_count += e - s

        self.indices = valid[~val_mask] if split == "train" else valid[val_mask]
        print(
            f"[{self.seq_dir.name}] Pose {split}: {len(self.indices)} samples "
            f"(frames: {self.total_frames}, N={n_frames}, offset={frame_offset})"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_voxel(self, idx: int) -> np.ndarray:
        with h5py.File(self._voxels_path, "r") as f:
            vox = f["voxels"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            vox = F.interpolate(
                torch.from_numpy(vox).unsqueeze(0),
                size=(rh, rw), mode="bilinear", align_corners=False,
            ).squeeze(0).numpy()
        return vox

    def _load_depth(self, idx: int) -> np.ndarray:
        with h5py.File(self._depth_path, "r") as f:
            depth = f["depth"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            depth = F.interpolate(
                torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                size=(rh, rw), mode="nearest",
            ).squeeze().numpy()
        return depth

    def _load_spatial_mask(self, idx: int) -> Optional[np.ndarray]:
        if not self._has_spatial_mask:
            return None
        with h5py.File(self._mask_path, "r") as f:
            sp = f["mask"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            sp = F.interpolate(
                torch.from_numpy(sp).unsqueeze(0).unsqueeze(0),
                size=(rh, rw), mode="nearest",
            ).squeeze().numpy()
        return sp

    def __getitem__(self, i: int):
        ref_global = int(self.indices[i])

        tgt_voxels = self._load_voxel(ref_global)   # (C, H, W)

        # Source voxels and T_source_from_target poses
        T_ref     = self._T_cam[ref_global]           # (4, 4)
        T_ref_inv = np.linalg.inv(T_ref)

        src_voxels_list: List[np.ndarray] = []
        T_s_from_t_list: List[np.ndarray] = []
        for offset in self.src_offsets:
            src_idx = ref_global + offset
            src_voxels_list.append(self._load_voxel(src_idx))
            # T_s_from_t = T_cam_world[src] @ inv(T_cam_world[ref])
            T_s_from_t_np = (self._T_cam[src_idx] @ T_ref_inv).astype(np.float32)
            T_s_from_t_list.append(T_s_from_t_np)

        depth = self._load_depth(ref_global)
        mask  = ((depth > self.depth_min) & (depth < self.depth_max)).astype(np.float32)
        sp    = self._load_spatial_mask(ref_global)
        if sp is not None:
            mask[sp == 0] = 0.0

        depth_norm = np.clip(
            (depth - self.depth_min) / (self.depth_max - self.depth_min), 0.0, 1.0
        ).astype(np.float32)

        return (
            torch.from_numpy(tgt_voxels),
            torch.from_numpy(np.stack(src_voxels_list).astype(np.float32)) if src_voxels_list
                else torch.zeros(0, *tgt_voxels.shape, dtype=torch.float32),     # (N_src, C, H, W)
            torch.from_numpy(depth_norm[None]),                                   # (1, H, W)
            torch.from_numpy(mask[None]),                                         # (1, H, W)
            torch.from_numpy(np.stack(T_s_from_t_list).astype(np.float32)) if T_s_from_t_list
                else torch.zeros(0, 4, 4, dtype=torch.float32),                  # (N_src, 4, 4)
        )


def build_pose_datasets(sequence_dirs: List[Path], args) -> Tuple[Dataset, Dataset]:
    """Build train/val RealPoseDataset from a list of sequence directories."""
    resize_hw = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize_hw = (args.resize_h, args.resize_w)

    train_datasets: List[Dataset] = []
    val_datasets:   List[Dataset] = []
    for sd in sequence_dirs:
        try:
            train_datasets.append(RealPoseDataset(
                sd, n_frames=args.n_frames, frame_offset=args.frame_offset,
                depth_min=args.depth_min, depth_max=args.depth_max,
                resize_hw=resize_hw, split="train", val_ratio=args.val_ratio,
            ))
            val_datasets.append(RealPoseDataset(
                sd, n_frames=args.n_frames, frame_offset=args.frame_offset,
                depth_min=args.depth_min, depth_max=args.depth_max,
                resize_hw=resize_hw, split="val", val_ratio=args.val_ratio,
            ))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Warning: skipping {sd} for pose training: {exc}")
    if not train_datasets:
        raise RuntimeError("No valid sequences for pose training.")
    train = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val   = ConcatDataset(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]
    return train, val


# ─────────────────────────────────────────────────────────────────────
# Single-frame dataset (stateless models such as UNet)
# ─────────────────────────────────────────────────────────────────────

class RealSingleDataset(Dataset):
    """Flat single-frame dataset for stateless models (e.g. UNet).

    Returns ``(voxels (C,H,W), depth_norm (1,H,W), mask (1,H,W))``.
    No sequence dimension, no source frames — one optimiser step per frame.
    """

    def __init__(
        self,
        sequence_dir,
        depth_min:  float = DEPTH_MIN,
        depth_max:  float = D_MAX,
        resize_hw:  Optional[Tuple[int, int]] = None,
        split:      str   = "train",
        val_ratio:  float = 0.1,
        seed:       int   = 42,
    ):
        self.seq_dir   = Path(sequence_dir)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.resize_hw = resize_hw

        self._voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self._depth_path  = self.seq_dir / "hdf5" / "depth_in_event_frame.h5"
        self._mask_path   = self.seq_dir / "hdf5" / "spatial_mask.h5"

        for p in (self._voxels_path, self._depth_path):
            if not p.exists():
                raise FileNotFoundError(p)
        self._has_spatial_mask = self._mask_path.exists()

        with h5py.File(self._depth_path, "r") as f:
            self.total_frames = f["depth"].shape[0]

        rng        = np.random.default_rng(seed)
        block_size = 20
        n_blocks   = max(1, (self.total_frames + block_size - 1) // block_size)
        block_ids  = np.arange(n_blocks)
        rng.shuffle(block_ids)
        n_val    = max(1, int(round(self.total_frames * val_ratio)))
        val_mask = np.zeros(self.total_frames, dtype=bool)
        val_count = 0
        for b in block_ids:
            if val_count >= n_val:
                break
            s = b * block_size
            e = min((b + 1) * block_size, self.total_frames)
            val_mask[s:e] = True
            val_count += e - s

        all_idx     = np.arange(self.total_frames, dtype=np.int64)
        self.indices = all_idx[~val_mask] if split == "train" else all_idx[val_mask]
        print(f"[{self.seq_dir.name}] Single {split}: {len(self.indices)} frames")

    def __len__(self) -> int:
        return len(self.indices)

    def _load_voxel(self, idx: int) -> np.ndarray:
        with h5py.File(self._voxels_path, "r") as f:
            v = f["voxels"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            v = F.interpolate(
                torch.from_numpy(v).unsqueeze(0),
                size=(rh, rw), mode="bilinear", align_corners=False,
            ).squeeze(0).numpy()
        return v

    def _load_depth(self, idx: int) -> np.ndarray:
        with h5py.File(self._depth_path, "r") as f:
            d = f["depth"][idx].astype(np.float32)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            d = F.interpolate(
                torch.from_numpy(d).unsqueeze(0).unsqueeze(0),
                size=(rh, rw), mode="nearest",
            ).squeeze().numpy()
        return d

    def __getitem__(self, i: int):
        idx   = int(self.indices[i])
        vox   = self._load_voxel(idx)                   # (C, H, W)
        depth = self._load_depth(idx)                   # (H, W)
        mask  = ((depth > self.depth_min) & (depth < self.depth_max)).astype(np.float32)
        if self._has_spatial_mask:
            with h5py.File(self._mask_path, "r") as f:
                sp = f["mask"][idx].astype(np.float32)
            if self.resize_hw is not None:
                rh, rw = self.resize_hw
                sp = F.interpolate(
                    torch.from_numpy(sp).unsqueeze(0).unsqueeze(0),
                    size=(rh, rw), mode="nearest",
                ).squeeze().numpy()
            mask[sp == 0] = 0.0
        depth_norm = np.clip(
            (depth - self.depth_min) / (self.depth_max - self.depth_min), 0.0, 1.0
        ).astype(np.float32)
        return (
            torch.from_numpy(vox),
            torch.from_numpy(depth_norm[None]),   # (1, H, W)
            torch.from_numpy(mask[None]),          # (1, H, W)
        )


def build_single_datasets(sequence_dirs: List[Path], args) -> Tuple[Dataset, Dataset]:
    """Build train/val RealSingleDataset from a list of sequence directories."""
    resize_hw = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize_hw = (args.resize_h, args.resize_w)
    trains, vals = [], []
    for sd in sequence_dirs:
        try:
            trains.append(RealSingleDataset(
                sd, depth_min=args.depth_min, depth_max=args.depth_max,
                resize_hw=resize_hw, split="train", val_ratio=args.val_ratio,
            ))
            vals.append(RealSingleDataset(
                sd, depth_min=args.depth_min, depth_max=args.depth_max,
                resize_hw=resize_hw, split="val", val_ratio=args.val_ratio,
            ))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Warning: skipping {sd} for single training: {exc}")
    if not trains:
        raise RuntimeError("No valid sequences for single-frame training.")
    train = ConcatDataset(trains) if len(trains) > 1 else trains[0]
    val   = ConcatDataset(vals)   if len(vals)   > 1 else vals[0]
    return train, val


# -----------------------------
# Training Utilities
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    K: Optional[torch.Tensor] = None,
    epoch: int = 0,
    log_interval: float = 10.0,
    profile: bool = False,
) -> Dict[str, float]:
    """Train for one epoch with sequence processing."""
    model.train()
    
    total_loss   = 0.0
    total_charb  = 0.0
    total_grad   = 0.0
    total_smooth = 0.0
    total_normal = 0.0
    total_mean_a = 0.0
    total_mv     = 0.0
    n_batches    = 0
    n_total = len(loader)

    # Per-phase timing accumulators (seconds) — only used when profile=True.
    # Phases: data loading → host-to-device transfer → forward+loss → backward+step
    t_data     = 0.0
    t_transfer = 0.0
    t_forward  = 0.0
    t_backward = 0.0

    last_log_time = time.time()
    epoch_start   = time.time()
    _t = time.perf_counter()  # start of the current data-loading window (reset each iteration)

    for events, depths, masks, poses in loader:
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_data += time.perf_counter() - _t

        # events: (B, T, C, H, W)
        # depths: (B, T, 1, H, W)
        # masks:  (B, T, 1, H, W)
        # poses:  (B, T, 4, 4) relative T_curr_from_prev per step
        B, T = events.shape[:2]

        if profile:
            _t2 = time.perf_counter()
        events = events.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        poses = poses.to(device, non_blocking=True)
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_transfer += time.perf_counter() - _t2

        # Process sequence
        states    = None
        pred_prev = None
        mask_prev = None
        batch_loss   = 0.0
        batch_charb  = 0.0
        batch_grad   = 0.0
        batch_smooth = 0.0
        batch_normal = 0.0
        batch_mean_a = 0.0
        batch_mv     = 0.0

        if profile:
            _t2 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            for t in range(T):
                T_rel = poses[:, t] if model.use_pose_warp else None
                pred, states = model(events[:, t], states, T_rel=T_rel)
                # Normal and multi-view consistency losses are expensive (they
                # require backprojection and re-projection).  We skip them on
                # odd timesteps — adjacent frames share nearly the same geometry,
                # so computing them every other step cuts cost ~50% with negligible
                # impact on training dynamics.
                K_step = K if t % 2 == 0 else None
                loss, metrics = model.compute_loss(
                    pred, depths[:, t], masks[:, t], events[:, t],
                    pred_prev=pred_prev,
                    T_curr_from_prev=poses[:, t] if t > 0 else None,
                    mask_prev=mask_prev,
                    K=K_step,
                )
                
                batch_loss   += loss
                batch_charb  += metrics["charb"]
                batch_grad   += metrics["grad"]
                batch_smooth += metrics["smooth"]
                batch_normal += metrics["normal"]
                batch_mean_a += metrics["mean"]
                batch_mv     += metrics["mv"]

                # Detach pred before passing it as the "previous" frame to the next
                # timestep's MV loss.  Gradients already flowed through pred at time t;
                # keeping it attached would create an unwanted computation graph cycle.
                pred_prev = pred.detach()
                mask_prev = masks[:, t]
        
        # Average over sequence
        batch_loss = batch_loss / T
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_forward += time.perf_counter() - _t2

        if profile:
            _t2 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(batch_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_backward += time.perf_counter() - _t2
        
        total_loss   += batch_loss.item()
        total_charb  += batch_charb  / T
        total_grad   += batch_grad   / T
        total_smooth += batch_smooth / T
        total_normal += batch_normal / T
        total_mean_a += batch_mean_a / T
        total_mv     += batch_mv     / T
        n_batches    += 1

        now = time.time()
        if now - last_log_time >= log_interval:
            elapsed = now - epoch_start
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0
            eta_sec = (n_total - n_batches) / batches_per_sec if batches_per_sec > 0 else 0
            avg_loss = total_loss / n_batches
            print(
                f"  [Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                f"| loss: {avg_loss:.5f} "
                f"| {batches_per_sec:.1f} batch/s "
                f"| ETA: {int(eta_sec // 60):02d}:{int(eta_sec % 60):02d}",
                flush=True,
            )
            last_log_time = now

        if profile:
            _t = time.perf_counter()  # reset for next data-loading window

    if profile and n_batches > 0:
        total_t = t_data + t_transfer + t_forward + t_backward
        def _pct(x): return 100.0 * x / total_t if total_t > 0 else 0.0
        def _ms(x): return 1000.0 * x / n_batches
        print(
            f"\n  [Epoch {epoch:03d}] Timing breakdown ({n_batches} batches):\n"
            f"    Data loading : {_ms(t_data):7.1f} ms/batch  ({_pct(t_data):.1f}%)\n"
            f"    Host→device  : {_ms(t_transfer):7.1f} ms/batch  ({_pct(t_transfer):.1f}%)\n"
            f"    Forward+loss : {_ms(t_forward):7.1f} ms/batch  ({_pct(t_forward):.1f}%)\n"
            f"    Backward+opt : {_ms(t_backward):7.1f} ms/batch  ({_pct(t_backward):.1f}%)\n"
            f"    Total        : {_ms(total_t):7.1f} ms/batch\n",
            flush=True,
        )

    n = max(1, n_batches)
    return {
        "total":  total_loss   / n,
        "charb":  total_charb  / n,
        "grad":   total_grad   / n,
        "smooth": total_smooth / n,
        "normal": total_normal / n,
        "mean":   total_mean_a / n,
        "mv":     total_mv     / n,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> Dict[str, float]:
    """Validate and compute metrics over all frames after recurrent warmup.

    Evaluates frames t >= eval_start = min(3, T-1) so the ConvLSTM state has
    had a few steps to warm up before measurements begin.  All four standard
    metrics are accumulated:
      - L1 (metric)  — mean absolute error in metres
      - AbsRel        — |pred-gt|/gt
      - RMSE          — sqrt(mean squared error) in metres
      - δ<1.25        — fraction with max(pred/gt, gt/pred) < 1.25
    """
    model.eval()

    total_l1      = 0.0
    total_abs_rel = 0.0
    total_sq      = 0.0
    total_delta    = 0.0
    total_pixels   = 0.0
    n_frames       = 0  # count each evaluated frame, not each batch

    sum_pred_m     = 0.0
    sum_gt_m       = 0.0
    sum_valid_mean = 0.0

    with torch.no_grad():
        for events, depths, masks, poses in loader:
            B, T = events.shape[:2]
            events = events.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            poses  = poses.to(device, non_blocking=True)

            # Warm up the ConvLSTM state for the first few frames before measuring.
            # Without warmup, the zero-initialised hidden state produces systematically
            # worse predictions that would bias the validation metrics downward.
            eval_start = min(3, T - 1)

            states = None
            preds  = []
            for t in range(T):
                T_rel = poses[:, t] if model.use_pose_warp else None
                pred, states = model(events[:, t], states, T_rel=T_rel)
                if t >= eval_start:
                    preds.append((pred, depths[:, t], masks[:, t]))

            for pred, gt, mask in preds:
                pred_metric = linear_normalized_to_depth(pred, d_min=depth_min, d_max=depth_max)
                gt_metric   = linear_normalized_to_depth(gt,   d_min=depth_min, d_max=depth_max)

                gt_safe  = gt_metric.clamp_min(1e-6)
                diff     = torch.abs(pred_metric - gt_metric) * mask
                n_valid  = mask.sum().clamp_min(1.0)

                total_l1      += (diff.sum() / n_valid).item()
                total_abs_rel += ((diff / gt_safe).sum() / n_valid).item()
                total_sq      += ((diff ** 2).sum() / n_valid).item()

                ratio = torch.max(pred_metric / gt_safe, gt_safe / pred_metric.clamp_min(1e-6))
                total_delta   += ((ratio < 1.25).float() * mask).sum().item()
                total_pixels  += n_valid.item()

                # Accumulate global mean stats (all frames, not just last batch)
                n_valid_pix     = mask.sum().item()
                sum_pred_m     += (pred_metric * mask).sum().item()
                sum_gt_m       += (gt_metric   * mask).sum().item()
                sum_valid_mean += n_valid_pix

                n_frames += 1  # one frame evaluated

    n = max(1, n_frames)
    _sv = max(1.0, sum_valid_mean)
    return {
        "l1_metric":        total_l1      / n,
        "abs_rel":          total_abs_rel / n,
        "rmse":             (total_sq     / n) ** 0.5,
        "delta_125":        total_delta   / max(1.0, total_pixels),
        "pred_metric_mean": sum_pred_m    / _sv,
        "gt_metric_mean":   sum_gt_m      / _sv,
    }


def _depth_to_rgb(t: torch.Tensor) -> torch.Tensor:
    """Apply turbo colormap to a (1, H, W) or (H, W) float tensor in [0, 1].

    Returns a (3, H, W) float32 tensor suitable for writer.add_image.
    """
    import matplotlib.cm as cm

    arr = t[0].cpu().float().numpy() if t.dim() == 3 else t.cpu().float().numpy()
    rgba = cm.turbo(arr)  # (H, W, 4) float64
    rgb = rgba[:, :, :3].transpose(2, 0, 1).astype("float32")  # (3, H, W)
    return torch.from_numpy(rgb)


def log_images(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    split: str = "val",
    n_images: int = 3,
    fixed_indices: Optional[List[int]] = None,
):
    """Log one composite row image per sample to TensorBoard under '<split>/sample_N'.

    Each row contains 8 panels side-by-side (all at the same H×W):
        rgb | events | mask | gt_depth | pred_depth | error | depth+events overlay | depth×mask

    Uses ``fixed_indices`` (determined once before training) so the same samples
    are shown every epoch, making progress directly comparable across epochs.
    RGB is loaded from ``rgb_in_event_frame.h5`` when available; otherwise the
    panel is left black.
    """
    from torch.utils.data import Subset

    model.eval()

    dataset = loader.dataset
    n = len(dataset)
    if fixed_indices is not None:
        indices = [i % n for i in fixed_indices[:n_images]]
    else:
        indices = torch.randperm(n)[:n_images].tolist()

    # Process one sample at a time so we can build per-sample row images.
    subset_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn if hasattr(loader, "collate_fn") and loader.collate_fn is not None else None,
    )

    def _resolve_dataset(ds, global_idx: int):
        """Return (RealDataset, local_idx) for a ConcatDataset or plain Dataset."""
        if isinstance(ds, ConcatDataset):
            for cumsum, sub_ds in zip(ds.cumulative_sizes, ds.datasets):
                if global_idx < cumsum:
                    return sub_ds, global_idx - (cumsum - len(sub_ds))
        return ds, global_idx

    def _load_rgb_panel(ds, global_idx: int, H: int, W: int) -> Optional[torch.Tensor]:
        """Load RGB from realsense.h5 for the given dataset sample index.
        Returns a (3, H, W) float32 tensor in [0, 1], or None if unavailable."""
        real_ds, local_idx = _resolve_dataset(ds, global_idx)
        seq_dir = getattr(real_ds, "sequence_dir", None)
        if seq_dir is None:
            return None
        rs_path = seq_dir / "hdf5" / "realsense.h5"
        if not rs_path.exists():
            return None
        try:
            with h5py.File(rs_path, "r") as _f:
                if "rgb" not in _f:
                    return None
                frame_indices = getattr(real_ds, "indices", None)
                frame_idx = int(frame_indices[local_idx]) if frame_indices is not None else local_idx
                rgb = _f["rgb"][frame_idx].astype(np.float32)  # (H_rs, W_rs, 3)
            # resize to match depth/event resolution
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H_rs, W_rs)
            if rgb_t.shape[2] != H or rgb_t.shape[3] != W:
                rgb_t = torch.nn.functional.interpolate(rgb_t, size=(H, W), mode="bilinear", align_corners=False)
            rgb_t = rgb_t[0] / 255.0  # (3, H, W) in [0, 1]
            return rgb_t.contiguous()
        except Exception:
            return None

    with torch.no_grad():
        for sample_i, (events, depths, masks, poses) in enumerate(subset_loader):
            B, T = events.shape[:2]
            events_gpu = events.to(device)
            depths_gpu = depths.to(device)
            masks_gpu  = masks.to(device)
            poses_gpu  = poses.to(device)

            # Run full sequence; keep last prediction
            states = None
            pred = None
            for t in range(T):
                T_rel = poses_gpu[:, t] if model.use_pose_warp else None
                pred, states = model(events_gpu[:, t], states, T_rel=T_rel)

            H_img, W_img = depths.shape[-2], depths.shape[-1]

            # RGB panel loaded from realsense.h5 (falls back to black if unavailable)
            global_idx = indices[sample_i]
            rgb_vis = _load_rgb_panel(dataset, global_idx, H_img, W_img)
            if rgb_vis is None:
                rgb_vis = torch.zeros(3, H_img, W_img)

            # Events: first 3 bins → RGB; replicate if fewer channels
            ev = events[0, -1]  # (C, H, W)
            ev_vis = ev[:3] if ev.shape[0] >= 3 else ev[0:1].expand(3, -1, -1)
            ev_vis = (ev_vis - ev_vis.min()) / (ev_vis.max() - ev_vis.min() + 1e-6)
            ev_vis = ev_vis.contiguous()

            mask_img = masks[0, -1]                          # (1, H, W)
            mask_vis = mask_img.expand(3, -1, -1).contiguous()  # greyscale → 3-ch

            gt_vis   = _depth_to_rgb(depths[0, -1] * mask_img)

            pred_cpu = pred[0].cpu().contiguous()            # (1, H, W)
            pred_vis = _depth_to_rgb(pred_cpu * mask_img)

            err = torch.abs(pred_cpu - depths[0, -1]) * mask_img
            err_vis = _depth_to_rgb(err / (err.max() + 1e-6))

            # --- Depth + Events overlay (plasma depth + red/blue alpha composite) ---
            import matplotlib.cm as _cm
            depth_np = depths[0, -1].squeeze().cpu().float().numpy()  # (H, W) in [0, 1]
            ev_accum_np = ev.float().sum(0).numpy()                    # (H, W)
            plasma_depth_rgb = _cm.plasma(np.clip(depth_np, 0, 1))[:, :, :3].astype("float32")
            ev_abs = np.abs(ev_accum_np)
            ev_scale = float(ev_abs.max()) + 1e-6
            ev_alpha = np.clip(ev_abs / ev_scale, 0, 1) * 0.7          # (H, W)
            ev_r = np.clip(ev_accum_np, 0, None) / ev_scale
            ev_b = np.clip(-ev_accum_np, 0, None) / ev_scale
            ev_rgb_np = np.stack([ev_r, np.zeros_like(ev_r), ev_b], axis=-1)  # (H, W, 3)
            ov_np = plasma_depth_rgb * (1 - ev_alpha[..., None]) + ev_rgb_np * ev_alpha[..., None]
            ov_vis = torch.from_numpy(ov_np.transpose(2, 0, 1))         # (3, H, W)

            # --- Depth × Mask overlay (plasma, masked) ---
            dm_np = _cm.plasma(np.clip(depth_np * mask_img.squeeze().cpu().numpy(), 0, 1))[:, :, :3].astype("float32")
            dm_vis = torch.from_numpy(dm_np.transpose(2, 0, 1))         # (3, H, W)

            # Compose row: rgb | events | mask | gt_depth | pred_depth | error | depth+ev overlay | depth×mask
            row = torch.cat([rgb_vis, ev_vis, mask_vis, gt_vis, pred_vis, err_vis, ov_vis, dm_vis], dim=2)
            writer.add_image(f"{split}/sample_{sample_i}", row, epoch)


# ----------------------------------------
# Pose training / validation / logging
# ----------------------------------------

def train_one_epoch_pose(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    K: Optional[torch.Tensor] = None,
    epoch: int = 0,
    log_interval: float = 10.0,
) -> Dict[str, float]:
    """Train PoseUNet for one epoch with photometric + GT + smoothness losses."""
    model.train()

    total_loss   = 0.0
    total_charb  = 0.0
    total_smooth = 0.0
    total_photo  = 0.0
    n_batches    = 0
    n_total      = len(loader)
    last_log     = time.time()
    epoch_start  = time.time()

    for tgt_voxels, src_voxels, depths, masks, T_s_from_t in loader:
        tgt_voxels = tgt_voxels.to(device, non_blocking=True)   # (B, C, H, W)
        src_voxels = src_voxels.to(device, non_blocking=True)   # (B, N_src, C, H, W)
        depths     = depths.to(device, non_blocking=True)       # (B, 1, H, W)
        masks      = masks.to(device, non_blocking=True)        # (B, 1, H, W)
        T_s_from_t = T_s_from_t.to(device, non_blocking=True)  # (B, N_src, 4, 4)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            pred, _ = model(tgt_voxels, src_voxels=src_voxels, T_s_from_t=T_s_from_t)
            loss, metrics = model.compute_loss(
                pred, depths, masks, tgt_voxels,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss   += loss.item()
        total_charb  += metrics["charb"]
        total_smooth += metrics["smooth"]
        total_photo  += metrics["photo"]
        n_batches    += 1

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - epoch_start
            bps = n_batches / elapsed if elapsed > 0 else 0
            eta = (n_total - n_batches) / bps if bps > 0 else 0
            print(
                f"  [Pose Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                f"| loss: {total_loss / n_batches:.5f} "
                f"charb: {total_charb / n_batches:.4f} "
                f"photo: {total_photo / n_batches:.4f} "
                f"smooth: {total_smooth / n_batches:.5f} "
                f"| {bps:.1f} batch/s ETA: {int(eta // 60):02d}:{int(eta % 60):02d}",
                flush=True,
            )
            last_log = now

    n = max(1, n_batches)
    return {
        "total":  total_loss   / n,
        "charb":  total_charb  / n,
        "smooth": total_smooth / n,
        "photo":  total_photo  / n,
        # Keys expected by shared logging path
        "grad":   0.0,
        "normal": 0.0,
        "mean":   0.0,
        "mv":     0.0,
    }


@torch.no_grad()
def validate_pose(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> Dict[str, float]:
    """Validate PoseUNet — only the target frame is used for inference."""
    model.eval()

    total_l1      = 0.0
    total_abs_rel = 0.0
    total_sq      = 0.0
    total_delta   = 0.0
    total_pixels  = 0.0
    sum_pred_m    = 0.0
    sum_gt_m      = 0.0
    sum_valid_m   = 0.0
    n_frames      = 0

    for tgt_voxels, src_voxels, depths, masks, T_s_from_t in loader:
        tgt_voxels = tgt_voxels.to(device, non_blocking=True)
        src_voxels = src_voxels.to(device, non_blocking=True)
        depths     = depths.to(device, non_blocking=True)
        masks      = masks.to(device, non_blocking=True)
        T_s_from_t = T_s_from_t.to(device, non_blocking=True)

        pred, _ = model(tgt_voxels, src_voxels=src_voxels, T_s_from_t=T_s_from_t)

        pred_metric = linear_normalized_to_depth(pred,   d_min=depth_min, d_max=depth_max)
        gt_metric   = linear_normalized_to_depth(depths, d_min=depth_min, d_max=depth_max)

        gt_safe = gt_metric.clamp_min(1e-6)
        diff    = torch.abs(pred_metric - gt_metric) * masks
        n_valid = masks.sum().clamp_min(1.0)

        total_l1      += (diff.sum() / n_valid).item()
        total_abs_rel += ((diff / gt_safe).sum() / n_valid).item()
        total_sq      += ((diff ** 2).sum() / n_valid).item()

        ratio = torch.max(pred_metric / gt_safe, gt_safe / pred_metric.clamp_min(1e-6))
        total_delta  += ((ratio < 1.25).float() * masks).sum().item()
        total_pixels += n_valid.item()

        n_valid_pix   = masks.sum().item()
        sum_pred_m   += (pred_metric * masks).sum().item()
        sum_gt_m     += (gt_metric   * masks).sum().item()
        sum_valid_m  += n_valid_pix
        n_frames     += 1

    n   = max(1, n_frames)
    _sv = max(1.0, sum_valid_m)
    return {
        "l1_metric":        total_l1      / n,
        "abs_rel":          total_abs_rel / n,
        "rmse":             (total_sq     / n) ** 0.5,
        "delta_125":        total_delta   / max(1.0, total_pixels),
        "pred_metric_mean": sum_pred_m    / _sv,
        "gt_metric_mean":   sum_gt_m      / _sv,
    }


@torch.no_grad()
def log_images_pose(
    writer: SummaryWriter,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epoch: int,
    split: str = "val",
    n_images: int = 3,
) -> None:
    """Log [rgb | events | mask | gt | init_depth | pred | error | overlay | depth×mask].

    init_depth is the stage-1 (target-only) prediction cached in
    ``model._init_depth_cache`` after the forward pass.
    """
    import matplotlib.cm as _cm

    def _load_rgb(ds, global_idx: int, H: int, W: int) -> Optional[torch.Tensor]:
        """Load RGB from realsense.h5; handles ConcatDataset transparently."""
        if isinstance(ds, ConcatDataset):
            for cumsum, sub_ds in zip(ds.cumulative_sizes, ds.datasets):
                if global_idx < cumsum:
                    ds         = sub_ds
                    global_idx = global_idx - (cumsum - len(sub_ds))
                    break
        seq_dir = getattr(ds, "seq_dir", getattr(ds, "sequence_dir", None))
        if seq_dir is None:
            return None
        rs_path = Path(seq_dir) / "hdf5" / "realsense.h5"
        if not rs_path.exists():
            return None
        try:
            with h5py.File(rs_path, "r") as _f:
                if "rgb" not in _f:
                    return None
                frame_indices = getattr(ds, "indices", None)
                frame_idx = int(frame_indices[global_idx]) if frame_indices is not None else global_idx
                rgb = _f["rgb"][frame_idx].astype(np.float32)
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            if rgb_t.shape[2] != H or rgb_t.shape[3] != W:
                rgb_t = F.interpolate(rgb_t, size=(H, W), mode="bilinear", align_corners=False)
            return (rgb_t[0] / 255.0).contiguous()
        except Exception:
            return None

    model.eval()
    n       = len(dataset)
    indices = np.linspace(0, n - 1, min(n_images, n), dtype=int).tolist()

    with torch.no_grad():
        for sample_i, idx in enumerate(indices):
            tgt_voxels, src_voxels, depth_norm, mask, T_s_from_t = dataset[idx]

            pred, _ = model(
                tgt_voxels.unsqueeze(0).to(device),
                src_voxels=src_voxels.unsqueeze(0).to(device),
                T_s_from_t=T_s_from_t.unsqueeze(0).to(device),
            )   # (1, 1, H, W)
            pred_cpu = pred[0].cpu()                                # (1, H, W)

            # Stage-1 init depth cached by the model's forward pass
            init_cache = getattr(model, "_init_depth_cache", None)
            init_cpu   = init_cache[0].cpu() if init_cache is not None else None

            H_img, W_img = depth_norm.shape[-2], depth_norm.shape[-1]

            # ── RGB ─────────────────────────────────────────────────────────
            rgb_vis = _load_rgb(dataset, idx, H_img, W_img)
            if rgb_vis is None:
                rgb_vis = torch.zeros(3, H_img, W_img)

            # ── Events ──────────────────────────────────────────────────────
            ev_gray = tgt_voxels.abs().sum(0)
            ev_vis  = (ev_gray / (ev_gray.max() + 1e-6)).unsqueeze(0).expand(3, -1, -1).contiguous()

            # ── Mask ────────────────────────────────────────────────────────
            mask_vis = mask.expand(3, -1, -1).contiguous()

            # ── Depth panels (turbo colormap via _depth_to_rgb) ─────────────
            gt_vis   = _depth_to_rgb(depth_norm * mask)
            pred_vis = _depth_to_rgb(pred_cpu   * mask)
            err      = torch.abs(pred_cpu - depth_norm) * mask
            err_vis  = _depth_to_rgb(err / (err.max() + 1e-6))
            init_vis = _depth_to_rgb(init_cpu * mask) if init_cpu is not None \
                       else torch.zeros(3, H_img, W_img)

            # ── Depth + events overlay (plasma + red/blue alpha) ─────────
            depth_np   = depth_norm.squeeze().float().numpy()
            ev_accum   = tgt_voxels.float().sum(0).numpy()
            plasma_rgb = _cm.plasma(np.clip(depth_np, 0, 1))[:, :, :3].astype("float32")
            ev_abs     = np.abs(ev_accum)
            ev_scale   = float(ev_abs.max()) + 1e-6
            ev_alpha   = np.clip(ev_abs / ev_scale, 0, 1) * 0.7
            ev_r       = np.clip( ev_accum, 0, None) / ev_scale
            ev_b       = np.clip(-ev_accum, 0, None) / ev_scale
            ev_rgb_np  = np.stack([ev_r, np.zeros_like(ev_r), ev_b], axis=-1)
            ov_np      = plasma_rgb * (1 - ev_alpha[..., None]) + ev_rgb_np * ev_alpha[..., None]
            ov_vis     = torch.from_numpy(ov_np.transpose(2, 0, 1))

            # ── Depth × mask (plasma) ────────────────────────────────────
            dm_np  = _cm.plasma(np.clip(depth_np * mask.squeeze().numpy(), 0, 1))[:, :, :3].astype("float32")
            dm_vis = torch.from_numpy(dm_np.transpose(2, 0, 1))

            # Row: rgb | events | mask | gt | init | pred | error | overlay | depth×mask
            row = torch.cat([rgb_vis, ev_vis, mask_vis, gt_vis, init_vis, pred_vis, err_vis, ov_vis, dm_vis], dim=2)
            writer.add_image(f"{split}/sample_{sample_i}", row, epoch)

    model.train()


# ─────────────────────────────────────────────────────────────────────
# Single-frame training / validation / logging (stateless models)
# ─────────────────────────────────────────────────────────────────────

def train_one_epoch_single(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int = 0,
    log_interval: float = 10.0,
) -> Dict[str, float]:
    """Train a stateless single-frame model for one epoch.

    Each sample is a standalone (voxels, depth, mask) triple — no sequence
    looping, no state management.  Significantly faster than the sequence
    path for models like UNet that have no recurrent units.
    """
    model.train()
    total_loss = total_charb = total_smooth = total_grad = 0.0
    total_normal = total_mean = total_mv = 0.0
    n_batches    = 0
    n_total      = len(loader)
    last_log     = time.time()
    epoch_start  = time.time()

    for voxels, gt, mask in loader:
        voxels = voxels.to(device, non_blocking=True)
        gt     = gt.to(device, non_blocking=True)
        mask   = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            pred, _ = model(voxels, None, None)
            loss, metrics = model.compute_loss(pred, gt, mask, voxels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss   += loss.item()
        total_charb  += metrics.get("charb",  0.0)
        total_smooth += metrics.get("smooth", 0.0)
        total_grad   += metrics.get("grad",   0.0)
        total_normal += metrics.get("normal", 0.0)
        total_mean   += metrics.get("mean",   0.0)
        total_mv     += metrics.get("mv",     0.0)
        n_batches    += 1

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - epoch_start
            bps = n_batches / elapsed if elapsed > 0 else 0
            eta = (n_total - n_batches) / bps if bps > 0 else 0
            print(
                f"  [Single Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                f"| loss: {total_loss / n_batches:.5f} "
                f"charb: {total_charb / n_batches:.4f} "
                f"smooth: {total_smooth / n_batches:.5f} "
                f"| {bps:.1f} batch/s ETA: {int(eta // 60):02d}:{int(eta % 60):02d}",
                flush=True,
            )
            last_log = now

    n = max(1, n_batches)
    return {
        "total":  total_loss   / n,
        "charb":  total_charb  / n,
        "smooth": total_smooth / n,
        "grad":   total_grad   / n,
        "normal": total_normal / n,
        "mean":   total_mean   / n,
        "mv":     total_mv     / n,
    }


@torch.no_grad()
def validate_single(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    depth_min: float = DEPTH_MIN,
    depth_max: float = D_MAX,
) -> Dict[str, float]:
    """Validate a stateless single-frame model."""
    model.eval()
    total_l1 = total_abs_rel = total_sq = total_delta = total_pixels = 0.0
    sum_pred_m = sum_gt_m = sum_valid_m = 0.0
    n_frames = 0

    for voxels, gt, mask in loader:
        voxels = voxels.to(device, non_blocking=True)
        gt     = gt.to(device, non_blocking=True)
        mask   = mask.to(device, non_blocking=True)

        pred, _ = model(voxels, None, None)

        pred_metric = linear_normalized_to_depth(pred, d_min=depth_min, d_max=depth_max)
        gt_metric   = linear_normalized_to_depth(gt,   d_min=depth_min, d_max=depth_max)
        gt_safe     = gt_metric.clamp_min(1e-6)
        diff        = torch.abs(pred_metric - gt_metric) * mask
        n_valid     = mask.sum().clamp_min(1.0)

        total_l1      += (diff.sum() / n_valid).item()
        total_abs_rel += ((diff / gt_safe).sum() / n_valid).item()
        total_sq      += ((diff ** 2).sum() / n_valid).item()
        ratio          = torch.max(pred_metric / gt_safe, gt_safe / pred_metric.clamp_min(1e-6))
        total_delta   += ((ratio < 1.25).float() * mask).sum().item()
        total_pixels  += n_valid.item()

        sum_pred_m  += (pred_metric * mask).sum().item()
        sum_gt_m    += (gt_metric   * mask).sum().item()
        sum_valid_m += mask.sum().item()
        n_frames    += 1

    n   = max(1, n_frames)
    _sv = max(1.0, sum_valid_m)
    return {
        "l1_metric":        total_l1      / n,
        "abs_rel":          total_abs_rel / n,
        "rmse":             (total_sq     / n) ** 0.5,
        "delta_125":        total_delta   / max(1.0, total_pixels),
        "pred_metric_mean": sum_pred_m    / _sv,
        "gt_metric_mean":   sum_gt_m      / _sv,
    }


@torch.no_grad()
def log_images_single(
    writer: SummaryWriter,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epoch: int,
    split: str = "val",
    n_images: int = 3,
) -> None:
    """Log [events | gt_depth | pred_depth | error] rows to TensorBoard."""
    model.eval()
    n       = len(dataset)
    indices = np.linspace(0, n - 1, min(n_images, n), dtype=int).tolist()

    for sample_i, idx in enumerate(indices):
        voxels, depth_norm, mask = dataset[idx]

        pred, _ = model(voxels.unsqueeze(0).to(device), None, None)  # (1, 1, H, W)
        pred_cpu = pred[0].cpu()                                       # (1, H, W)

        ev_gray = voxels.abs().sum(0)
        ev_vis  = (ev_gray / (ev_gray.max() + 1e-6)).unsqueeze(0).expand(3, -1, -1)

        gt_m = linear_normalized_to_depth(depth_norm.unsqueeze(0), d_min=model.depth_min, d_max=model.depth_max)[0]
        pr_m = linear_normalized_to_depth(pred_cpu.unsqueeze(0),   d_min=model.depth_min, d_max=model.depth_max)[0]

        valid_d = gt_m[0][mask[0] > 0.5]
        vmin = float(valid_d.min()) if valid_d.numel() > 0 else 0.0
        vmax = float(valid_d.max()) if valid_d.numel() > 0 else 1.0

        def _cmap(d):
            d_n = ((d - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1).numpy()
            r = np.clip(1.5 - np.abs(d_n * 4.0 - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(d_n * 4.0 - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(d_n * 4.0 - 1.0), 0.0, 1.0)
            return torch.from_numpy(np.stack([r, g, b]).astype(np.float32))

        gt_vis   = _cmap(gt_m[0] * mask[0])
        pred_vis = _cmap(pr_m[0] * mask[0])
        err_vis  = _cmap((torch.abs(pr_m[0] - gt_m[0]) * mask[0]) / (vmax - vmin + 0.05))

        row = torch.cat([ev_vis, gt_vis, pred_vis, err_vis], dim=2)
        writer.add_image(f"{split}/sample_{sample_i}", row, epoch)

    model.train()


# -----------------------------------
# MVS training / validation / logging
# -----------------------------------

def train_one_epoch_magnet(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_interval: float = 10.0,
) -> Dict[str, float]:
    """Train MaGNet for one epoch using the NLL loss with geometric weighting."""
    from models.magnet import magnet_nll_loss
    model.train()
    total_loss = total_err = 0.0
    n_batches  = 0
    n_total    = len(loader)
    last_log   = time.time()
    epoch_start = time.time()

    for batch in loader:
        imgs         = batch["imgs"].to(device, non_blocking=True)
        proj_mats    = batch["proj_mats"].to(device, non_blocking=True)
        depth_values = batch["depth_values"].to(device, non_blocking=True)
        depth_gt     = batch["depth"].to(device, non_blocking=True)
        mask         = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            depth_est, _sigma, all_preds = model(imgs, proj_mats, depth_values)

            # Downsample GT and mask to the model's coarse output resolution (H/4 × W/4)
            h, w = depth_est.shape[-2:]
            depth_gt_ds = F.interpolate(
                depth_gt.unsqueeze(1), size=(h, w), mode="nearest"
            ).squeeze(1)
            mask_ds = F.interpolate(
                mask.unsqueeze(1), size=(h, w), mode="nearest"
            ).squeeze(1)

            loss = magnet_nll_loss(all_preds, depth_gt_ds, mask_ds, gamma=model.gamma)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            valid = mask_ds > 0.5
            err = torch.abs(depth_est[valid] - depth_gt_ds[valid]).mean().item() if valid.any() else 0.0

        total_loss += loss.item()
        total_err  += err
        n_batches  += 1

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - epoch_start
            bps = n_batches / elapsed if elapsed > 0 else 0
            eta = (n_total - n_batches) / bps if bps > 0 else 0
            print(
                f"  [MaGNet Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                f"| loss: {total_loss / n_batches:.5f} abs_err: {total_err / n_batches:.4f}m "
                f"| {bps:.1f} batch/s ETA: {int(eta // 60):02d}:{int(eta % 60):02d}",
                flush=True,
            )
            last_log = now

    n = max(1, n_batches)
    return {"total": total_loss / n, "abs_err": total_err / n}


def train_one_epoch_mvs(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_interval: float = 10.0,
) -> Dict[str, float]:
    from models.mvsnet import mvsnet_loss, abs_depth_error
    model.train()
    total_loss = total_err = 0.0
    n_batches  = 0
    n_total    = len(loader)
    last_log   = time.time()
    epoch_start = time.time()

    for batch in loader:
        imgs         = batch["imgs"].to(device, non_blocking=True)
        proj_mats    = batch["proj_mats"].to(device, non_blocking=True)
        depth_values = batch["depth_values"].to(device, non_blocking=True)
        depth_gt     = batch["depth"].to(device, non_blocking=True)
        mask         = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            depth_est, _, _ = model(imgs, proj_mats, depth_values)
            depth_gt_ds = F.interpolate(
                depth_gt.unsqueeze(1), size=depth_est.shape[-2:], mode="nearest"
            ).squeeze(1)
            mask_ds = F.interpolate(
                mask.unsqueeze(1), size=depth_est.shape[-2:], mode="nearest"
            ).squeeze(1)
            loss = mvsnet_loss(depth_est, depth_gt_ds, mask_ds)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        err = abs_depth_error(depth_est.detach(), depth_gt_ds, mask_ds).item()
        total_loss += loss.item()
        total_err  += err
        n_batches  += 1

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - epoch_start
            bps = n_batches / elapsed if elapsed > 0 else 0
            eta = (n_total - n_batches) / bps if bps > 0 else 0
            print(
                f"  [MVS Epoch {epoch:03d}] {n_batches}/{n_total} batches "
                f"| loss: {total_loss / n_batches:.5f} abs_err: {total_err / n_batches:.4f}m "
                f"| {bps:.1f} batch/s ETA: {int(eta // 60):02d}:{int(eta % 60):02d}",
                flush=True,
            )
            last_log = now

    n = max(1, n_batches)
    return {"total": total_loss / n, "abs_err": total_err / n}


@torch.no_grad()
def validate_mvs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate MVSNet and return metrics evaluated at **full resolution**.

    MVSNet outputs depth at ~1/4 of the input resolution.  To make the L1 /
    AbsRel / RMSE / δ<1.25 numbers directly comparable to the e2depth metrics
    (which are computed at full resolution), we upsample the estimate back to
    the ground-truth resolution before measuring error.  Training loss is still
    computed at the model's native output resolution (unchanged).
    """
    from models.mvsnet import mvsnet_loss
    model.eval()
    total_loss    = 0.0
    total_l1      = 0.0
    total_abs_rel = 0.0
    total_sq      = 0.0
    total_delta   = 0.0
    total_pixels  = 0.0
    n_batches     = 0

    for batch in loader:
        imgs         = batch["imgs"].to(device, non_blocking=True)
        proj_mats    = batch["proj_mats"].to(device, non_blocking=True)
        depth_values = batch["depth_values"].to(device, non_blocking=True)
        depth_gt     = batch["depth"].to(device, non_blocking=True)   # (B, H, W) metres
        mask         = batch["mask"].to(device, non_blocking=True)     # (B, H, W)

        depth_est, _, _ = model(imgs, proj_mats, depth_values)         # (B, h, w) coarse

        # Training loss at native output resolution
        depth_gt_ds = F.interpolate(
            depth_gt.unsqueeze(1), size=depth_est.shape[-2:], mode="nearest"
        ).squeeze(1)
        mask_ds = F.interpolate(
            mask.unsqueeze(1), size=depth_est.shape[-2:], mode="nearest"
        ).squeeze(1)
        total_loss += mvsnet_loss(depth_est, depth_gt_ds, mask_ds).item()

        # Upsample estimate to full resolution for fair metric comparison
        depth_up = F.interpolate(
            depth_est.unsqueeze(1), size=depth_gt.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)                                                    # (B, H, W)

        gt_safe = depth_gt.clamp_min(1e-6)
        diff    = torch.abs(depth_up - depth_gt) * mask
        n_valid = mask.sum().clamp_min(1.0)

        total_l1      += (diff.sum() / n_valid).item()
        total_abs_rel += ((diff / gt_safe).sum() / n_valid).item()
        total_sq      += ((diff ** 2).sum() / n_valid).item()

        ratio = torch.max(depth_up / gt_safe,
                          gt_safe / depth_up.clamp_min(1e-6))
        total_delta  += ((ratio < 1.25).float() * mask).sum().item()
        total_pixels += n_valid.item()

        n_batches += 1

    n = max(1, n_batches)
    return {
        "l1_metric": total_l1      / n,
        "abs_rel":   total_abs_rel / n,
        "rmse":      (total_sq     / n) ** 0.5,
        "delta_125": total_delta   / max(1.0, total_pixels),
        "abs_err":   total_l1      / n,   # backward-compat alias
        "total":     total_loss    / n,
    }


@torch.no_grad()
def log_images_mvs(
    writer: SummaryWriter,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epoch: int,
    split: str = "val",
    n_images: int = 3,
) -> None:
    """Log [events | gt_depth | pred_depth | error] rows to TensorBoard for MVSNet."""
    model.eval()
    n = len(dataset)
    indices = np.linspace(0, n - 1, min(n_images, n), dtype=int).tolist()

    for sample_i, idx in enumerate(indices):
        sample       = dataset[idx]
        imgs         = sample["imgs"].unsqueeze(0).to(device)
        proj_mats    = sample["proj_mats"].unsqueeze(0).to(device)
        depth_values = sample["depth_values"].unsqueeze(0).to(device)
        depth_gt     = sample["depth"]   # (H, W)
        mask         = sample["mask"]    # (H, W)

        depth_est, _, _ = model(imgs, proj_mats, depth_values)  # (1, h, w)
        depth_est_up = F.interpolate(
            depth_est.cpu().unsqueeze(1), size=depth_gt.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze()  # (H, W)

        vox     = sample["imgs"][0]                           # ref view (C, H, W)
        ev_gray = vox.abs().sum(0)
        ev_vis  = (ev_gray / (ev_gray.max() + 1e-6)).unsqueeze(0).expand(3, -1, -1)

        gt_masked   = depth_gt * mask
        pred_masked = depth_est_up * mask
        valid_d     = gt_masked[gt_masked > 0]
        vmin = float(valid_d.min()) if valid_d.numel() > 0 else 0.0
        vmax = float(valid_d.max()) if valid_d.numel() > 0 else 1.0

        def _cmap(d):
            d_n = ((d - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1).numpy()
            r = np.clip(1.5 - np.abs(d_n * 4.0 - 3.0), 0.0, 1.0)
            g = np.clip(1.5 - np.abs(d_n * 4.0 - 2.0), 0.0, 1.0)
            b = np.clip(1.5 - np.abs(d_n * 4.0 - 1.0), 0.0, 1.0)
            return torch.from_numpy(np.stack([r, g, b]).astype(np.float32))

        err_vis = _cmap((torch.abs(depth_est_up - depth_gt) * mask) / (vmax - vmin + 0.05))
        row = torch.cat([ev_vis, _cmap(gt_masked), _cmap(pred_masked), err_vis], dim=2)
        writer.add_image(f"{split}/sample_{sample_i}", row, epoch)

    model.train()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train E2Depth (recurrent UNet) for event-to-depth prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_dir", nargs="+", type=str, default=None,
                           help="Sequence directory(ies) to train on")
    data_group.add_argument("--data_root", type=str, default=str(DATA_ROOT),
                           help="Root data directory (will use all subdirs)")
    data_group.add_argument("--val_ratio", type=float, default=0.2,
                           help="Fraction of data for validation (object-level split when >1 sequence, temporal otherwise)")
    data_group.add_argument("--num_bins", type=int, default=5,
                           help="Number of temporal bins for voxel grid")
    data_group.add_argument("--depth_max", type=float, default=D_MAX,
                           help="Maximum depth in meters")
    data_group.add_argument("--depth_min", type=float, default=DEPTH_MIN,
                           help="Minimum depth in meters")
    data_group.add_argument("--use_pose_warp", action=argparse.BooleanOptionalAction, default=True,
                           help="Warp ConvLSTM hidden states with relative camera pose (from poses.h5)")
    data_group.add_argument("--spatial_mask", action="store_true", default=True,
                           help="Mask pixels outside cube around EE (from spatial_mask.h5, "
                                "run precompute_spatial_mask.py first)")
    # Model – architecture selection; model-specific args are added below via MODEL_ARG_REGISTRY
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default="e2depth",
                            choices=list(MODEL_REGISTRY.keys()),
                            help="Model architecture to train")

    # Pre-parse --model so we can let the chosen model register its own arguments.
    _pre, _ = parser.parse_known_args()
    MODEL_ARG_REGISTRY[_pre.model](model_group)

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=25,
                            help="Number of epochs")
    train_group.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE,
                            help="Batch size")
    train_group.add_argument("--seq_len", type=int, default=TRAIN_SEQ_LEN,
                            help="Sequence length for recurrent training")
    train_group.add_argument("--frame_stride", type=int, default=1,
                            help="Step between frames in each sequence (1=consecutive, k=skip k-1 frames)")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="Peak learning rate")
    train_group.add_argument("--lr_min", type=float, default=1e-6,
                            help="Minimum LR at end of cosine decay")
    train_group.add_argument("--warmup_epochs", type=int, default=0,
                            help="Linear warmup epochs (0 = no warmup)")
    train_group.add_argument("--lambda_grad", type=float, default=0.5,
                            help="Weight for multi-scale gradient loss")
    train_group.add_argument("--lambda_smooth", type=float, default=0.01,
                            help="Weight for edge-aware smoothness loss")
    train_group.add_argument("--lambda_normal", type=float, default=0.1,
                            help="Weight for surface normal cosine loss")
    train_group.add_argument("--lambda_mean", type=float, default=0.1,
                            help="Weight for global mean alignment loss")
    train_group.add_argument("--lambda_mv", type=float, default=0.2,
                            help="Weight for multi-view consistency loss (needs poses)")
    train_group.add_argument("--num_workers", type=int, default=8)
    train_group.add_argument("--crop_h", type=int, default=TRAIN_CROP_HW[0],
                            help="Center crop height in pixels (0=no crop)")
    train_group.add_argument("--crop_w", type=int, default=TRAIN_CROP_HW[1],
                            help="Center crop width in pixels (0=no crop)")
    train_group.add_argument("--resize_h", type=int, default=TRAIN_RESIZE_HW[0],
                            help="Resize height before crop/augmentation (0=no resize)")
    train_group.add_argument("--resize_w", type=int, default=TRAIN_RESIZE_HW[1],
                            help="Resize width before crop/augmentation (0=no resize)")
    
    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--out_dir", type=str, default=None,
                          help="Checkpoint/log output directory (default: training/checkpoints/<model>)")
    out_group.add_argument("--save_every", type=int, default=10,
                          help="Save checkpoint every N epochs")
    out_group.add_argument("--resume", type=str, default=None,
                          help="Path to checkpoint to resume from")

    # Debug
    dbg_group = parser.add_argument_group("Debug")
    dbg_group.add_argument("--debug_viz", action="store_true",
                           help="Generate a debug PNG of sample inputs and exit (no training)")
    dbg_group.add_argument("--debug_n", type=int, default=3,
                           help="Number of random timestamps shown in debug PNG")
    dbg_group.add_argument("--debug_out", type=str, default="debug_viz.png",
                           help="Output path for the debug PNG")
    dbg_group.add_argument("--debug_seed", type=int, default=None,
                           help="Random seed for timestamp sampling in debug mode")
    dbg_group.add_argument("--debug_seq", type=str, default=None,
                           help="Specific sequence directory to visualize "
                                "(default: first found via --data_dir / --data_root)")
    dbg_group.add_argument("--profile", action="store_true",
                           help="Print per-phase timing breakdown (data/transfer/forward/backward) "
                                "at the end of each training epoch")

    args = parser.parse_args()

    # Default output directory: training/checkpoints/<model_name>
    if args.out_dir is None:
        args.out_dir = str(Path(__file__).resolve().parent / "checkpoints" / args.model)

    # Find sequences
    if args.data_dir:
        sequence_dirs = [Path(d) for d in args.data_dir]
    else:
        sequence_dirs = find_sequence_dirs(Path(args.data_root))
        if not sequence_dirs:
            print(f"No valid sequences found in {args.data_root}")
            print("Expected: <dir>/hdf5/realsense.h5 + <dir>/events/voxels_cam0/")
            return
    
    print(f"Found {len(sequence_dirs)} sequences:")
    for d in sequence_dirs:
        print(f"  - {d.name}")

    # Debug visualization mode — generate PNG and exit
    if args.debug_viz:
        if args.debug_seq:
            dbg_seq = Path(args.debug_seq)
        else:
            dbg_seq = sequence_dirs[0]
        dbg_resize_hw = None
        if args.resize_h > 0 and args.resize_w > 0:
            dbg_resize_hw = (args.resize_h, args.resize_w)
        dbg_crop_size = None
        if args.crop_h > 0 and args.crop_w > 0:
            dbg_crop_size = (args.crop_h, args.crop_w)
        debug_visualize(
            str(dbg_seq),
            n_samples=args.debug_n,
            out_path=args.debug_out,
            num_bins=args.num_bins,
            seed=args.debug_seed,
            resize_hw=dbg_resize_hw,
            crop_size=dbg_crop_size,
            use_pose=False,  # pose depth channel visualization removed (now using hidden-state warp)
            spatial_mask=args.spatial_mask,
        )
        return

    # Determine model data mode and build datasets
    data_type = MODEL_DATA_TYPE.get(args.model, "sequence")

    # Models that don't support pose warp — disable it so poses aren't loaded.
    _NO_POSE_WARP_MODELS = {"unet", "mvsnet", "magnet", "pose_unet"}
    if args.model in _NO_POSE_WARP_MODELS:
        args.use_pose_warp = False

    crop_size = None
    resize_hw = None
    if data_type == "single":
        if args.resize_h > 0 and args.resize_w > 0:
            resize_hw = (args.resize_h, args.resize_w)
        train_ds, val_ds = build_single_datasets(sequence_dirs, args)
    elif data_type == "pose":
        if args.resize_h > 0 and args.resize_w > 0:
            resize_hw = (args.resize_h, args.resize_w)
        train_ds, val_ds = build_pose_datasets(sequence_dirs, args)
    elif data_type in ("mvs", "magnet"):
        if args.resize_h > 0 and args.resize_w > 0:
            resize_hw = (args.resize_h, args.resize_w)
        train_ds, val_ds = build_mvs_datasets(sequence_dirs, args)
    else:
        if args.crop_h > 0 and args.crop_w > 0:
            crop_size = (args.crop_h, args.crop_w)
        if args.resize_h > 0 and args.resize_w > 0:
            resize_hw = (args.resize_h, args.resize_w)

        cfg = DataConfig(
            seq_len=args.seq_len,
            crop_size=crop_size,
            resize_hw=resize_hw,
            depth_max=args.depth_max,
            depth_min=args.depth_min,
            augment=False,
            num_bins=args.num_bins,
            use_pose_warp=args.use_pose_warp,
            spatial_mask=args.spatial_mask,
            frame_stride=args.frame_stride,
        )

        # Create config without augmentation for validation
        cfg_val = DataConfig(
            seq_len=args.seq_len,
            crop_size=crop_size,
            resize_hw=resize_hw,
            depth_max=args.depth_max,
            depth_min=args.depth_min,
            augment=False,
            num_bins=args.num_bins,
            use_pose_warp=args.use_pose_warp,
            spatial_mask=args.spatial_mask,
            frame_stride=args.frame_stride,
        )

        # Datasets — object-level split when multiple sequences are available,
        # fallback to temporal block-split within the single sequence.
        if len(sequence_dirs) > 1:
            rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 42)
            dirs_shuffled = list(sequence_dirs)
            rng.shuffle(dirs_shuffled)
            n_val_dirs = max(1, int(round(len(dirs_shuffled) * args.val_ratio)))
            val_dirs = dirs_shuffled[:n_val_dirs]
            train_dirs = dirs_shuffled[n_val_dirs:]
            if not train_dirs:
                # Edge case: only 1 dir total, fall through to temporal split
                train_dirs = val_dirs
            print(f"  Object-level split: {len(train_dirs)} train dirs, {len(val_dirs)} val dirs")
            train_ds = create_multi_sequence_dataset(
                [str(d) for d in train_dirs], cfg, split="train", val_ratio=0.0
            )
            val_ds = create_multi_sequence_dataset(
                [str(d) for d in val_dirs], cfg_val, split="val", val_ratio=1.0
            )
        else:
            # Single sequence: temporal block-split within that sequence
            train_ds = create_multi_sequence_dataset(
                [str(d) for d in sequence_dirs], cfg, split="train", val_ratio=args.val_ratio
            )
            val_ds = create_multi_sequence_dataset(
                [str(d) for d in sequence_dirs], cfg_val, split="val", val_ratio=args.val_ratio
            )

    # Print dataset info
    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Sequences: {len(sequence_dirs)}")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Valid samples: {len(val_ds)}")
    print(f"  Depth range: {args.depth_min:.2f}m - {args.depth_max:.2f}m")
    if data_type == "sequence":
        print(f"  Depth encoding: linear")
        print(f"  Voxel bins: {args.num_bins}")
        print(f"  Pose warp: {args.use_pose_warp}")
    elif data_type == "single":
        print(f"  Voxel bins: {args.num_bins}")
    elif data_type == "pose":
        print(f"  N frames: {args.n_frames}")
        print(f"  Frame offset: {args.frame_offset}")
        print(f"  lambda_gt: {args.lambda_gt}  lambda_aux: {args.lambda_aux}  lambda_smooth: {args.lambda_smooth}")
    else:
        print(f"  Num views: {args.num_views}")
        print(f"  Num depth hypotheses: {args.num_depth}")
        print(f"  View interval: {args.view_interval}")
    print(f"{'='*60}\n")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Input channels = num_bins (pose is no longer an extra input channel)
    in_channels = args.num_bins
    print(f"Input channels: {in_channels} (bins={args.num_bins})")

    # Compute effective camera intrinsics K at the actual network input resolution.
    # K is calibrated at the native event-camera resolution; we propagate the pixel
    # coordinate transforms applied by resize and center-crop:
    #   resize:      fx *= rw/W,  fy *= rh/H,  cx *= rw/W,  cy *= rh/H
    #   center-crop: cx -= x0,    cy -= y0      (top-left corner of crop)
    # K is required for the normal loss, multi-view consistency loss, and pose warp.
    K_input: Optional[np.ndarray] = None
    input_hw: Optional[Tuple[int, int]] = None
    try:
        K_native  = np.load(str(CALIB_DIR / "event_intrinsics.npz"))["camera_matrix"].copy()
        native_ev = np.load(str(CALIB_DIR / "event_intrinsics.npz"))["image_size"]
        native_H, native_W = int(native_ev[1]), int(native_ev[0])
        K_input = K_native.copy()
        cur_H, cur_W = native_H, native_W
        if resize_hw is not None:
            rh, rw = resize_hw
            K_input[0] *= rw / native_W   # scale fx and cx by width ratio
            K_input[1] *= rh / native_H   # scale fy and cy by height ratio
            cur_H, cur_W = rh, rw
        if crop_size is not None:
            ch, cw = crop_size
            y0 = (cur_H - ch) // 2        # top-left row of the center crop
            x0 = (cur_W - cw) // 2        # top-left col of the center crop
            K_input[0, 2] -= x0           # shift cx by crop offset
            K_input[1, 2] -= y0           # shift cy by crop offset
            cur_H, cur_W = ch, cw
        input_hw = (cur_H, cur_W)
        print(f"Effective K (input res {cur_W}x{cur_H}):\n{K_input}")
    except Exception as e:
        print(f"Warning: could not load camera intrinsics: {e}")
        if data_type == "sequence" and args.use_pose_warp:
            raise

    # Model
    build_fn = MODEL_REGISTRY[args.model]
    model = build_fn(args, in_channels, K_input, input_hw).to(device)
    print(f"Model: {args.model}")

    # channels_last: faster conv on Ampere+ GPUs (NHWC layout)
    # Only apply for sequence models — MVSNet uses 3D convolutions which require channels_last_3d
    if data_type in ("sequence", "single", "pose"):
        model = model.to(memory_format=torch.channels_last)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    # LR schedule: optional linear warmup followed by cosine annealing.
    #   warmup  (epochs 1 … warmup_epochs):  LR rises linearly from ~0 → lr
    #   cosine  (remaining epochs):           LR decays as cosine from lr → lr_min
    # If warmup_epochs=0 (default) the cosine schedule starts from epoch 1.
    _cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_cosine_epochs, eta_min=args.lr_min
    )
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6 / args.lr,  # start very close to 0
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    start_epoch = 1
    best_val_loss    = float("inf")
    best_val_metrics: Optional[Dict[str, float]] = None
    best_val_epoch   = -1

    # Pick fixed dataset indices for TensorBoard image logging.
    # Using the same indices every epoch makes visual progress easy to compare.
    import json
    _rng_viz = np.random.default_rng(42)
    n_val   = len(val_loader.dataset)
    n_train = len(train_loader.dataset)
    viz_val_indices   = _rng_viz.choice(n_val,   size=min(3, n_val),   replace=False).tolist()
    viz_train_indices = _rng_viz.choice(n_train, size=min(3, n_train), replace=False).tolist()
    viz_indices_path = os.path.join(args.out_dir, "viz_indices.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(viz_indices_path, "w") as _f:
        json.dump({"val": viz_val_indices, "train": viz_train_indices}, _f)

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}")
    
    # TensorBoard
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.out_dir, "runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        if data_type == "single":
            train_metrics = train_one_epoch_single(
                model, train_loader, optimizer, scaler, device, epoch=epoch,
            )
            val_metrics = validate_single(model, val_loader, device,
                                          depth_min=args.depth_min,
                                          depth_max=args.depth_max)
        elif data_type == "pose":
            train_metrics = train_one_epoch_pose(
                model, train_loader, optimizer, scaler, device,
                K=torch.from_numpy(K_input).float() if K_input is not None else None,
                epoch=epoch,
            )
            val_metrics = validate_pose(model, val_loader, device,
                                        depth_min=args.depth_min,
                                        depth_max=args.depth_max)
        elif data_type == "magnet":
            train_metrics = train_one_epoch_magnet(
                model, train_loader, optimizer, scaler, device, epoch
            )
            val_metrics = validate_mvs(model, val_loader, device)
        elif data_type == "mvs":
            train_metrics = train_one_epoch_mvs(
                model, train_loader, optimizer, scaler, device, epoch
            )
            val_metrics = validate_mvs(model, val_loader, device)
        else:
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device,
                K=torch.from_numpy(K_input).float() if K_input is not None else None,
                epoch=epoch,
                profile=args.profile,
            )
            val_metrics = validate(model, val_loader, device,
                                   depth_min=args.depth_min,
                                   depth_max=args.depth_max)

        # Step cosine/warmup scheduler once per epoch
        scheduler.step()

        # Logging
        writer.add_scalar("loss/train_total", train_metrics["total"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if data_type in ("sequence", "single", "pose"):
            writer.add_scalar("loss/train_charb",   train_metrics["charb"],      epoch)
            writer.add_scalar("loss/train_grad",    train_metrics["grad"],       epoch)
            writer.add_scalar("loss/train_smooth",  train_metrics["smooth"],     epoch)
            writer.add_scalar("loss/train_normal",  train_metrics["normal"],     epoch)
            writer.add_scalar("loss/train_mean",    train_metrics["mean"],       epoch)
            writer.add_scalar("loss/train_mv",      train_metrics["mv"],         epoch)
            writer.add_scalar("loss/val_l1_metric", val_metrics["l1_metric"],    epoch)
            writer.add_scalar("loss/val_abs_rel",   val_metrics["abs_rel"],      epoch)
            writer.add_scalar("loss/val_rmse",      val_metrics["rmse"],         epoch)
            writer.add_scalar("loss/val_delta125",  val_metrics["delta_125"],    epoch)
        else:
            writer.add_scalar("loss/train_abs_err", train_metrics["abs_err"],    epoch)
            writer.add_scalar("loss/val_l1_metric", val_metrics["l1_metric"],    epoch)
            writer.add_scalar("loss/val_abs_rel",   val_metrics["abs_rel"],      epoch)
            writer.add_scalar("loss/val_rmse",      val_metrics["rmse"],         epoch)
            writer.add_scalar("loss/val_delta125",  val_metrics["delta_125"],    epoch)

        # GPU stats
        gpu_stats = get_gpu_stats(device)
        if gpu_stats:
            writer.add_scalar("gpu/vram_used_mb",     gpu_stats["vram_used_mb"],     epoch)
            writer.add_scalar("gpu/vram_reserved_mb", gpu_stats["vram_reserved_mb"], epoch)

        if data_type in ("sequence", "single", "pose"):
            print(f"Epoch {epoch:03d} | train: {train_metrics['total']:.5f} "
                  f"(c:{train_metrics['charb']:.4f} g:{train_metrics['grad']:.4f} "
                  f"s:{train_metrics['smooth']:.5f} n:{train_metrics['normal']:.4f} "
                  f"m:{train_metrics['mean']:.5f} mv:{train_metrics['mv']:.4f}) "
                  f"| val L1: {val_metrics['l1_metric']:.4f} | AbsRel: {val_metrics['abs_rel']:.4f} "
                  f"| RMSE: {val_metrics['rmse']:.4f} | δ<1.25: {val_metrics['delta_125']:.3f}")
            print(f"          | pred_m: {val_metrics['pred_metric_mean']:.3f}m vs gt_m: {val_metrics['gt_metric_mean']:.3f}m")
        else:
            print(f"Epoch {epoch:03d} | train loss: {train_metrics['total']:.5f} "
                  f"abs_err: {train_metrics['abs_err']:.4f}m "
                  f"| val L1: {val_metrics['l1_metric']:.4f}m "
                  f"| AbsRel: {val_metrics['abs_rel']:.4f} "
                  f"| RMSE: {val_metrics['rmse']:.4f}m "
                  f"| delta<1.25: {val_metrics['delta_125']:.3f}")
        if gpu_stats:
            print(f"          | VRAM: {gpu_stats['vram_used_mb']:.0f}/{gpu_stats['vram_reserved_mb']:.0f} MB (used/reserved)")
        
        # Overwrite best.pt whenever validation L1 improves
        if val_metrics["l1_metric"] < best_val_loss:
            best_val_loss    = val_metrics["l1_metric"]
            best_val_metrics = dict(val_metrics)
            best_val_epoch   = epoch
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {
                    "model": args.model,
                    "in_channels": in_channels,
                    "depth_max": args.depth_max,
                    "depth_min": args.depth_min,
                    **({"base": args.base, "num_encoders": args.num_encoders,
                        "num_residuals": args.num_residuals, "use_pose_warp": args.use_pose_warp}
                       if data_type == "sequence" else
                       {"base": args.base, "num_encoders": args.num_encoders,
                        "num_residuals": args.num_residuals}
                       if data_type == "single" else
                       {"n_frames": args.n_frames, "frame_offset": args.frame_offset,
                        "lambda_aux": args.lambda_aux}
                       if data_type == "pose" else
                       {"num_views": args.num_views, "num_depth": args.num_depth,
                        "view_interval": args.view_interval}),
                },
            }, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> Saved best model (val L1: {best_val_loss:.4f})")
        
        # Periodic snapshot (keeps training recoverable even if best.pt is stale)
        if epoch % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"))
        
        # Log composite prediction images to TensorBoard for both splits
        if data_type == "sequence":
            log_images(writer, model, val_loader,   device, epoch, split="val",   n_images=3, fixed_indices=viz_val_indices)
            log_images(writer, model, train_loader, device, epoch, split="train", n_images=3, fixed_indices=viz_train_indices)
        elif data_type == "single":
            log_images_single(writer, model, val_ds,   device, epoch, split="val",   n_images=3)
            log_images_single(writer, model, train_ds, device, epoch, split="train", n_images=3)
        elif data_type == "pose":
            log_images_pose(writer, model, val_ds,   device, epoch, split="val",   n_images=3)
            log_images_pose(writer, model, train_ds, device, epoch, split="train", n_images=3)
        else:  # "mvs" or "magnet"
            log_images_mvs(writer, model, val_ds,   device, epoch, split="val",   n_images=3)
            log_images_mvs(writer, model, train_ds, device, epoch, split="train", n_images=3)
        writer.flush()  # ensure images are written to disk immediately
    
    writer.close()

    # ── Final metrics summary (for cross-run comparison) ────────────
    if best_val_metrics is not None:
        import json
        summary = {
            "model":      args.model,
            "out_dir":    args.out_dir,
            "best_epoch": best_val_epoch,
            "epochs":     args.epochs,
            "val": {
                "l1_metric_m": round(best_val_metrics["l1_metric"], 6),
                "abs_rel":     round(best_val_metrics["abs_rel"],   6),
                "rmse_m":      round(best_val_metrics["rmse"],      6),
                "delta_125":   round(best_val_metrics["delta_125"], 6),
            },
            "config": vars(args),
        }
        metrics_path = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_path, "w") as _f:
            json.dump(summary, _f, indent=2)

        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  BEST VALIDATION METRICS  (epoch {best_val_epoch:03d})")
        print(sep)
        print(f"  MAE  (L1)  : {best_val_metrics['l1_metric']:.4f} m")
        print(f"  AbsRel     : {best_val_metrics['abs_rel']:.4f}")
        print(f"  RMSE       : {best_val_metrics['rmse']:.4f} m")
        print(f"  delta<1.25 : {best_val_metrics['delta_125']:.4f}")
        print(sep)
        print(f"  Saved to: {metrics_path}")
        print(sep)
    else:
        print(f"\nTraining complete! Best val L1: {best_val_loss:.4f}")
    print(f"Checkpoints: {args.out_dir}")


if __name__ == "__main__":
    main()
