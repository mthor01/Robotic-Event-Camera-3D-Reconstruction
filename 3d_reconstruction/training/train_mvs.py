import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.utils as tv_utils
from torch.utils.tensorboard import SummaryWriter

from reconstruction_config import (
    DEPTH_MIN,
    D_MAX,
    CALIB_DIR as _CALIB_DIR,
    DATA_ROOT as _DATA_ROOT,
    TRAIN_CROP_HW,
    NUM_BINS,
)

CALIB_DIR = Path(__file__).resolve().parent / _CALIB_DIR


# ----------------------------
# Utils
# ----------------------------

def make_depth_values(depth_min: float, depth_interval: float, num_depth: int) -> np.ndarray:
    return depth_min + np.arange(num_depth, dtype=np.float32) * depth_interval


def load_calibration(calib_dir: Path) -> dict:
    """Load event-camera intrinsics and the T_event_from_ee transform."""
    ev = np.load(calib_dir / "event_intrinsics.npz")
    K_event = ev["camera_matrix"].astype(np.float32)      # (3, 3)
    ev_size = ev["image_size"]                             # [W, H]

    T_rgb_from_ee   = np.load(calib_dir / "T_rgb_from_ee.npz")["T"].astype(np.float32)
    T_event_from_rgb = np.load(calib_dir / "T_event_from_rgb.npz")["T"].astype(np.float32)
    T_event_from_ee  = T_event_from_rgb @ T_rgb_from_ee   # (4, 4)

    return {
        "K_event": K_event,
        "ev_w": int(ev_size[0]),
        "ev_h": int(ev_size[1]),
        "T_event_from_ee": T_event_from_ee,
    }


def compute_proj_mat(
    K: np.ndarray,
    T_cam_from_world: np.ndarray,
    orig_hw: Tuple[int, int],
    target_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Compute scaled 3×4 projection matrix K_scaled @ [R | t]."""
    K = K.copy()
    if target_hw is not None:
        orig_h, orig_w = orig_hw
        new_h, new_w = target_hw
        K[0] *= new_w / orig_w   # scale fx, cx
        K[1] *= new_h / orig_h   # scale fy, cy
    return (K @ T_cam_from_world[:3, :]).astype(np.float32)  # (3, 4)


def colorize_depth(depth: torch.Tensor, vmin: float = None, vmax: float = None) -> torch.Tensor:
    """Convert (H, W) depth tensor to (3, H, W) float RGB via a jet-like colormap."""
    d = depth.float().cpu().numpy()
    valid = d > 0
    if vmin is None:
        vmin = float(d[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(d[valid].max()) if valid.any() else 1.0
    d_n = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(d_n * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(d_n * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(d_n * 4.0 - 1.0), 0.0, 1.0)
    return torch.from_numpy(np.stack([r, g, b], axis=0).astype(np.float32))


def collect_viz_batch(dataset: Dataset, n: int = 4) -> dict:
    """Return a batched dict of *n* evenly-spaced samples from *dataset*."""
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int).tolist()
    samples = [dataset[i] for i in indices]
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


# ----------------------------
# Dataset
# ----------------------------

class RealMVSDataset(Dataset):
    """
    MVSNet dataset built from real robot-recording data.

    Expected layout per sequence directory::

        <seq_dir>/
            hdf5/
                depth_in_event_frame.h5  # (N, H, W) float32 metres
                poses.h5                 # ee_T: (N, 4, 4) float64
                spatial_mask.h5          # mask: (N, H, W) uint8 {0,1}
            events/
                voxels_cam0.h5           # voxels: (N, NUM_BINS, H, W) float32

    Multi-view groups are formed by taking a reference frame and
    ``num_views - 1`` source frames sampled at ``±view_interval`` steps.
    Camera projection matrices are derived from robot EE poses and the
    event-camera intrinsics/extrinsics from ``calib_dir``.
    """

    def __init__(
        self,
        sequence_dir: str,
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
        self.seq_dir = Path(sequence_dir)
        self.calib = calib
        self.num_views = num_views
        self.num_depth = num_depth
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.resize_hw = resize_hw
        self.view_interval = view_interval
        self.split = split

        # depth_interval for the hypotheses plane sweep
        self.depth_interval = (depth_max - depth_min) / num_depth

        # Paths
        hdf5 = self.seq_dir / "hdf5"
        self._voxels_path = self.seq_dir / "events" / "voxels_cam0.h5"
        self._depth_path  = hdf5 / "depth_in_event_frame.h5"
        self._pose_path   = hdf5 / "poses.h5"
        self._mask_path   = hdf5 / "spatial_mask.h5"

        for p in (self._voxels_path, self._depth_path, self._pose_path):
            if not p.exists():
                raise FileNotFoundError(p)
        self._has_spatial_mask = self._mask_path.exists()

        # Load all poses into memory (small: N×4×4 float64)
        with h5py.File(self._pose_path, "r") as pf:
            ee_T = pf["ee_T"][:].astype(np.float32)  # (N, 4, 4) T_base_from_ee
        T_event_from_ee = calib["T_event_from_ee"]   # (4, 4)
        # T_cam_from_world[i] = T_event_from_ee @ inv(ee_T[i])
        T_ee_inv = np.linalg.inv(ee_T)               # (N, 4, 4)
        self._T_cam = np.einsum("ij,njk->nik", T_event_from_ee, T_ee_inv).astype(np.float32)

        with h5py.File(self._depth_path, "r") as df:
            self.n_frames = df["depth"].shape[0]
            self._orig_h  = df["depth"].shape[1]
            self._orig_w  = df["depth"].shape[2]

        # Source-frame offsets: symmetric around reference
        n_src = num_views - 1
        half  = n_src // 2
        offsets = []
        for k in range(1, half + 1):
            offsets += [k * view_interval, -k * view_interval]
        if n_src % 2 == 1:
            offsets.append((half + 1) * view_interval)
        self._src_offsets = offsets[:n_src]

        # Valid reference frames: all source frames must be in [0, n_frames)
        max_abs_offset = max(abs(o) for o in self._src_offsets) if self._src_offsets else 0
        valid = np.arange(max_abs_offset, self.n_frames - max_abs_offset, dtype=np.int64)

        # Train / val block split
        rng = np.random.default_rng(seed)
        block_size = max(10, max_abs_offset * 2 + 1)
        n_blocks = max(1, (len(valid) + block_size - 1) // block_size)
        block_ids = np.arange(n_blocks)
        rng.shuffle(block_ids)
        n_val = max(1, int(round(len(valid) * val_ratio)))
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
            f"[{self.seq_dir.name}] {split}: {len(self.indices)} samples "
            f"(frames: {self.n_frames}, views: {num_views}, "
            f"interval: {view_interval}, spatial_mask: {self._has_spatial_mask})"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_voxel(self, idx: int) -> np.ndarray:
        """Return (NUM_BINS, H, W) float32 event voxel grid at target resolution."""
        with h5py.File(self._voxels_path, "r") as f:
            vox = f["voxels"][idx].astype(np.float32)  # (NUM_BINS, H, W)
        assert vox.shape[0] == NUM_BINS, (
            f"{self._voxels_path}: voxels have {vox.shape[0]} bins but NUM_BINS={NUM_BINS}. "
            "Re-run precompute_voxels.py with --num_bins matching reconstruction_config.NUM_BINS."
        )
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            vox_t = torch.from_numpy(vox).unsqueeze(0)  # (1, NUM_BINS, H, W)
            vox = F.interpolate(vox_t, size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0).numpy()
        return vox

    def _load_depth(self, idx: int) -> np.ndarray:
        """Return (H, W) float32 metres at target resolution."""
        with h5py.File(self._depth_path, "r") as f:
            depth = f["depth"][idx].astype(np.float32)  # (H, W)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            depth = F.interpolate(depth_t, size=(rh, rw), mode="nearest").squeeze().numpy()
        return depth

    def _load_spatial_mask(self, idx: int) -> Optional[np.ndarray]:
        """Return (H, W) uint8 {0,1} at target resolution, or None."""
        if not self._has_spatial_mask:
            return None
        with h5py.File(self._mask_path, "r") as f:
            sp = f["mask"][idx].astype(np.float32)  # (H, W)
        if self.resize_hw is not None:
            rh, rw = self.resize_hw
            sp_t = torch.from_numpy(sp).unsqueeze(0).unsqueeze(0)
            sp = F.interpolate(sp_t, size=(rh, rw), mode="nearest").squeeze().numpy()
        return sp

    def __getitem__(self, i: int):
        ref_idx = int(self.indices[i])
        view_ids = [ref_idx] + [ref_idx + o for o in self._src_offsets]

        calib        = self.calib
        orig_hw      = (self._orig_h, self._orig_w)
        target_hw    = self.resize_hw

        imgs      = []
        proj_mats = []

        for vid in view_ids:
            img  = self._load_voxel(vid)                       # (NUM_BINS, H, W)
            proj = compute_proj_mat(
                calib["K_event"], self._T_cam[vid], orig_hw, target_hw
            )                                                  # (3, 4)
            imgs.append(img)
            proj_mats.append(proj)

        depth_values = make_depth_values(self.depth_min, self.depth_interval, self.num_depth)

        # Reference depth and validity mask
        depth = self._load_depth(ref_idx)                      # (H, W) metres
        mask  = (
            (depth > self.depth_min) & (depth < self.depth_max)
        ).astype(np.float32)

        sp = self._load_spatial_mask(ref_idx)
        if sp is not None:
            mask[sp == 0] = 0.0

        imgs        = np.stack(imgs).astype(np.float32)        # (V, NUM_BINS, H, W)
        proj_mats   = np.stack(proj_mats).astype(np.float32)  # (V, 3, 4)
        depth_values = depth_values.astype(np.float32)        # (D,)

        return {
            "imgs":         torch.from_numpy(imgs),
            "proj_mats":    torch.from_numpy(proj_mats),
            "depth_values": torch.from_numpy(depth_values),
            "depth":        torch.from_numpy(depth),
            "mask":         torch.from_numpy(mask),
        }


def find_sequence_dirs(data_root: Path) -> List[Path]:
    """Find valid sequence directories (must have the key HDF5 files)."""
    dirs = []
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        if (
            (d / "events" / "voxels_cam0.h5").exists()
            and (d / "hdf5" / "depth_in_event_frame.h5").exists()
            and (d / "hdf5" / "poses.h5").exists()
        ):
            dirs.append(d)
    return dirs


def build_dataset(
    sequence_dirs: List[Path],
    calib: dict,
    split: str,
    args,
) -> Dataset:
    resize_hw = (args.height, args.width)
    datasets = []
    for sd in sequence_dirs:
        try:
            ds = RealMVSDataset(
                sd,
                calib,
                num_views=args.num_views,
                num_depth=args.num_depth,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                resize_hw=resize_hw,
                view_interval=args.view_interval,
                split=split,
                val_ratio=args.val_ratio,
            )
            datasets.append(ds)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Warning: skipping {sd}: {exc}")
    if not datasets:
        raise RuntimeError(f"No valid sequences found for split '{split}'.")
    return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


# ----------------------------
# Model
# ----------------------------

class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            ConvBnReLU(NUM_BINS, 8),
            ConvBnReLU(8, 8),

            ConvBnReLU(8, 16, s=2),
            ConvBnReLU(16, 16),
            ConvBnReLU(16, 16),

            ConvBnReLU(16, 32, s=2),
            ConvBnReLU(32, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class Conv3dBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CostRegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = Conv3dBnReLU(32, 8)

        self.conv1 = Conv3dBnReLU(8, 16, s=2)
        self.conv2 = Conv3dBnReLU(16, 16)

        self.conv3 = Conv3dBnReLU(16, 32, s=2)
        self.conv4 = Conv3dBnReLU(32, 32)

        self.conv5 = Conv3dBnReLU(32, 64, s=2)
        self.conv6 = Conv3dBnReLU(64, 64)

        self.deconv7 = nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1)
        self.deconv9 = nn.ConvTranspose3d(32, 16, 3, 2, 1, output_padding=1)
        self.deconv11 = nn.ConvTranspose3d(16, 8, 3, 2, 1, output_padding=1)

        self.prob = nn.Conv3d(8, 1, 3, 1, 1)

    def forward(self, x):
        conv0 = self.conv0(x)

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))

        x = F.relu(F.interpolate(self.deconv7(x), size=conv4.shape[2:], mode="trilinear", align_corners=False) + conv4, inplace=True)
        x = F.relu(F.interpolate(self.deconv9(x), size=conv2.shape[2:], mode="trilinear", align_corners=False) + conv2, inplace=True)
        x = F.relu(F.interpolate(self.deconv11(x), size=conv0.shape[2:], mode="trilinear", align_corners=False) + conv0, inplace=True)

        x = self.prob(x)
        return x.squeeze(1)


def homo_warping(src_feat, src_proj, ref_proj, depth_values):
    """
    src_feat:     [B, C, H, W]
    src_proj:     [B, 3, 4]
    ref_proj:     [B, 3, 4]
    depth_values: [B, D]
    """

    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device

    src_proj_4 = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    ref_proj_4 = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

    src_proj_4[:, :3, :] = src_proj
    ref_proj_4[:, :3, :] = ref_proj

    proj = src_proj_4 @ torch.linalg.inv(ref_proj_4)
    rot = proj[:, :3, :3]
    trans = proj[:, :3, 3:4]

    y, x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.float32, device=device),
        torch.arange(0, W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    xyz = torch.stack((x, y, torch.ones_like(x)), dim=0)
    xyz = xyz.view(3, -1).unsqueeze(0).repeat(B, 1, 1)

    rot_xyz = rot @ xyz
    rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(B, 1, D, 1)
    proj_xyz = rot_depth_xyz + trans.view(B, 3, 1, 1)

    proj_xy = proj_xyz[:, :2] / proj_xyz[:, 2:3].clamp(min=1e-6)

    proj_x_normalized = proj_xy[:, 0] / ((W - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1] / ((H - 1) / 2) - 1

    grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)
    grid = grid.view(B, D, H * W, 2)

    warped = F.grid_sample(
        src_feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    warped = warped.view(B, C, D, H, W)
    return warped


def depth_regression(prob_volume, depth_values):
    return torch.sum(prob_volume * depth_values[:, :, None, None], dim=1)


class MVSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = FeatureNet()
        self.cost_reg = CostRegNet()

    def forward(self, imgs, proj_mats, depth_values):
        """
        imgs:         [B, V, NUM_BINS, H, W]
        proj_mats:    [B, V, 3, 4]
        depth_values: [B, D]
        """

        B, V, _, H, W = imgs.shape

        imgs = imgs.view(B * V, NUM_BINS, H, W)
        feats = self.feature(imgs)
        _, C, h, w = feats.shape
        feats = feats.view(B, V, C, h, w)

        ref_feat = feats[:, 0]
        src_feats = feats[:, 1:]

        ref_proj = proj_mats[:, 0].clone()
        src_projs = proj_mats[:, 1:].clone()

        ref_proj[:, :2, :] /= 4.0
        src_projs[:, :, :2, :] /= 4.0

        ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, depth_values.shape[1], 1, 1)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        for i in range(V - 1):
            warped = homo_warping(src_feats[:, i], src_projs[:, i], ref_proj, depth_values)
            volume_sum = volume_sum + warped
            volume_sq_sum = volume_sq_sum + warped ** 2

        volume_variance = volume_sq_sum / V - (volume_sum / V) ** 2

        cost = self.cost_reg(volume_variance)
        prob_volume = F.softmax(cost, dim=1)

        depth = depth_regression(prob_volume, depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]

        return depth, prob_volume, photometric_confidence


# ----------------------------
# Loss / Metrics
# ----------------------------

def mvsnet_loss(depth_est, depth_gt, mask):
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=depth_est.device, requires_grad=True)

    return F.smooth_l1_loss(depth_est[valid], depth_gt[valid])


@torch.no_grad()
def abs_depth_error(depth_est, depth_gt, mask):
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=depth_est.device)

    return torch.mean(torch.abs(depth_est[valid] - depth_gt[valid]))


# ----------------------------
# TensorBoard image logging
# ----------------------------

@torch.no_grad()
def log_images(
    model: nn.Module,
    batch: dict,
    writer: SummaryWriter,
    epoch: int,
    tag: str,
    device: torch.device,
) -> None:
    """Log a [ref-RGB | GT-depth | pred-depth] grid to TensorBoard."""
    was_training = model.training
    model.eval()

    imgs         = batch["imgs"].to(device)          # (N, V, NUM_BINS, H, W)
    proj_mats    = batch["proj_mats"].to(device)
    depth_values = batch["depth_values"].to(device)
    depth_gt     = batch["depth"].to(device)         # (N, H, W)
    mask         = batch["mask"].to(device)          # (N, H, W) float {0,1}

    depth_est, _, _ = model(imgs, proj_mats, depth_values)  # (N, h, w)
    depth_est_up = F.interpolate(
        depth_est.unsqueeze(1),
        size=depth_gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # (N, H, W)

    panels = []
    for b in range(imgs.shape[0]):
        # Collapse event bins → grayscale by summing absolute values, then replicate to 3ch
        vox      = imgs[b, 0].cpu()                           # (NUM_BINS, H, W)
        ev_gray  = vox.abs().sum(dim=0)                       # (H, W)
        ev_max   = ev_gray.max()
        ev_vis   = (ev_gray / (ev_max + 1e-6)).unsqueeze(0).expand(3, -1, -1)  # (3, H, W)
        m        = mask[b].cpu()                              # (H, W) float {0,1}
        gt_d     = depth_gt[b].cpu() * m                     # zero out invalid pixels
        pred_d   = depth_est_up[b].cpu() * m                 # zero out invalid pixels
        valid    = gt_d[gt_d > 0]
        vmin     = float(valid.min()) if valid.numel() > 0 else 0.0
        vmax     = float(valid.max()) if valid.numel() > 0 else 1.0
        err_d    = torch.abs(depth_est_up[b].cpu() - depth_gt[b].cpu()) * m
        panels += [ev_vis, colorize_depth(gt_d, vmin, vmax), colorize_depth(pred_d, vmin, vmax),
                   colorize_depth(err_d, 0.0, max(vmax - vmin, 0.05))]

    grid = tv_utils.make_grid(panels, nrow=4, padding=2, pad_value=0.5)
    writer.add_image(tag, grid, epoch)
    model.train(was_training)


# ----------------------------
# Train / Validate
# ----------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args, writer=None):
    model.train()

    total_loss = 0.0
    total_err = 0.0

    for step, batch in enumerate(loader):
        imgs = batch["imgs"].to(device)
        proj_mats = batch["proj_mats"].to(device)
        depth_values = batch["depth_values"].to(device)
        depth_gt = batch["depth"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=args.amp):
            depth_est, _, _ = model(imgs, proj_mats, depth_values)

            depth_gt_ds = F.interpolate(
                depth_gt.unsqueeze(1),
                size=depth_est.shape[-2:],
                mode="nearest",
            ).squeeze(1)

            mask_ds = F.interpolate(
                mask.unsqueeze(1),
                size=depth_est.shape[-2:],
                mode="nearest",
            ).squeeze(1)

            loss = mvsnet_loss(depth_est, depth_gt_ds, mask_ds)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        err = abs_depth_error(depth_est.detach(), depth_gt_ds, mask_ds)

        total_loss += loss.item()
        total_err += err.item()

        if step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} | step {step:05d}/{len(loader)} | "
                f"loss {loss.item():.4f} | abs_err {err.item():.4f}"
            )
            if writer is not None:
                global_step = (epoch - 1) * len(loader) + step
                writer.add_scalar("train/loss_step",    loss.item(), global_step)
                writer.add_scalar("train/abs_err_step", err.item(),  global_step)

    return total_loss / len(loader), total_err / len(loader)


@torch.no_grad()
def validate(model, loader, device, args, writer=None, epoch=None):
    model.eval()

    total_loss = 0.0
    total_err = 0.0

    for batch in loader:
        imgs = batch["imgs"].to(device)
        proj_mats = batch["proj_mats"].to(device)
        depth_values = batch["depth_values"].to(device)
        depth_gt = batch["depth"].to(device)
        mask = batch["mask"].to(device)

        depth_est, _, _ = model(imgs, proj_mats, depth_values)

        depth_gt_ds = F.interpolate(
            depth_gt.unsqueeze(1),
            size=depth_est.shape[-2:],
            mode="nearest",
        ).squeeze(1)

        mask_ds = F.interpolate(
            mask.unsqueeze(1),
            size=depth_est.shape[-2:],
            mode="nearest",
        ).squeeze(1)

        loss = mvsnet_loss(depth_est, depth_gt_ds, mask_ds)
        err = abs_depth_error(depth_est, depth_gt_ds, mask_ds)

        total_loss += loss.item()
        total_err += err.item()

    val_loss = total_loss / len(loader)
    val_err  = total_err / len(loader)
    if writer is not None and epoch is not None:
        writer.add_scalar("val/loss",    val_loss, epoch)
        writer.add_scalar("val/abs_err", val_err,  epoch)
    return val_loss, val_err


# ----------------------------
# Main
# ----------------------------

def save_checkpoint(path, model, optimizer, epoch, best_val):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train MVSNet on real robot-recorded data."
    )

    # --- Data ---
    p.add_argument(
        "--data_root", type=str, default=str(_DATA_ROOT),
        help="Root directory that contains per-object sequence folders.",
    )
    p.add_argument(
        "--data_dirs", type=str, nargs="+", default=None,
        help="Explicit list of sequence directories (overrides --data_root).",
    )
    p.add_argument(
        "--calib_dir", type=str, default=str(CALIB_DIR),
        help="Directory with calibration .npz files.",
    )
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument(
        "--view_interval", type=int, default=5,
        help="Frame-index step between reference and each source view.",
    )
    p.add_argument(
        "--depth_min", type=float, default=DEPTH_MIN,
        help="Near depth hypothesis plane (metres).",
    )
    p.add_argument(
        "--depth_max", type=float, default=D_MAX,
        help="Far depth hypothesis plane (metres).",
    )

    # --- Model / training ---
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_views", type=int, default=5)
    p.add_argument("--num_depth", type=int, default=192)
    p.add_argument("--height", type=int, default=TRAIN_CROP_HW[0])
    p.add_argument("--width",  type=int, default=TRAIN_CROP_HW[1])

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_dir", type=str, default="./checkpoints_mvs")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--amp", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    run_dir = os.path.join(args.save_dir, "runs")
    writer = SummaryWriter(log_dir=run_dir)
    print(f"TensorBoard: tensorboard --logdir {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Calibration ---
    calib = load_calibration(Path(args.calib_dir))

    # --- Sequence directories ---
    if args.data_dirs is not None:
        seq_dirs = [Path(d) for d in args.data_dirs]
    else:
        seq_dirs = find_sequence_dirs(Path(args.data_root))
    if not seq_dirs:
        raise RuntimeError(
            f"No valid sequence directories found under '{args.data_root}'. "
            "Each directory must contain events/voxels_cam0.h5, "
            "hdf5/depth_in_event_frame.h5, and hdf5/poses.h5."
        )
    print(f"Found {len(seq_dirs)} sequence(s).")

    train_set = build_dataset(seq_dirs, calib, "train", args)
    val_set   = build_dataset(seq_dirs, calib, "val",   args)

    # Fixed visualization batches (4 evenly-spaced samples per split)
    print("Collecting visualization batches...")
    viz_train_batch = collect_viz_batch(train_set, n=4)
    viz_val_batch   = collect_viz_batch(val_set,   n=4)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MVSNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 12, 14],
        gamma=0.5,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    best_val = float("inf")

    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from epoch {start_epoch}, best val {best_val:.4f}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_err = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args, writer=writer
        )

        val_loss, val_err = validate(model, val_loader, device, args, writer=writer, epoch=epoch)

        writer.add_scalar("train/loss",    train_loss, epoch)
        writer.add_scalar("train/abs_err", train_err,  epoch)

        log_images(model, viz_train_batch, writer, epoch, "train/depth_viz", device)
        log_images(model, viz_val_batch,   writer, epoch, "val/depth_viz",   device)

        scheduler.step()

        print(
            f"epoch {epoch:03d} done | "
            f"train loss {train_loss:.4f} | train abs_err {train_err:.4f} | "
            f"val loss {val_loss:.4f} | val abs_err {val_err:.4f}"
        )

        save_checkpoint(
            os.path.join(args.save_dir, "last.ckpt"),
            model,
            optimizer,
            epoch,
            best_val,
        )

        if val_err < best_val:
            best_val = val_err
            save_checkpoint(
                os.path.join(args.save_dir, "best.ckpt"),
                model,
                optimizer,
                epoch,
                best_val,
            )
            print(f"saved best checkpoint with val abs_err {best_val:.4f}")

    writer.close()


if __name__ == "__main__":
    main()