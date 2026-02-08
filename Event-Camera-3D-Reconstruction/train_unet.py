# python3 -m tensorboard.main --logdir checkpoints_event2depth/runs

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ================= DEFAULT DATA PATHS =================
DATA_DIR = Path("data") / "hdf5"

DEFAULT_EVENTS_CAM0 = DATA_DIR / "events_cam0.h5"
DEFAULT_EVENTS_CAM1 = DATA_DIR / "events_cam1.h5"
DEFAULT_REALSENSE   = DATA_DIR / "realsense.h5"

DEFAULT_OUT_DIR = Path("checkpoints_event2depth")
# =====================================================



# -----------------------------
# UNet (simple + solid baseline)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base: int = 32, out_channels: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# -----------------------------
# Losses (masked)
# -----------------------------
def masked_charbonnier(pred, gt, mask, eps=1e-3):
    # pred, gt, mask: (B,1,H,W)
    diff = (pred - gt) * mask
    loss = torch.sqrt(diff * diff + eps * eps)
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


def edge_aware_smoothness(pred, ref, mask):
    # pred/ref/mask: (B,1,H,W)
    def grad_x(t): return t[:, :, :, 1:] - t[:, :, :, :-1]
    def grad_y(t): return t[:, :, 1:, :] - t[:, :, :-1, :]

    pred = pred * mask
    ref = ref * mask

    px = grad_x(pred); py = grad_y(pred)
    rx = torch.abs(grad_x(ref)); ry = torch.abs(grad_y(ref))

    wx = torch.exp(-rx)
    wy = torch.exp(-ry)

    mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
    my = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_x = (torch.abs(px) * wx * mx).sum() / mx.sum().clamp_min(1.0)
    loss_y = (torch.abs(py) * wy * my).sum() / my.sum().clamp_min(1.0)
    return loss_x + loss_y


def log_images(writer, model, loader, device, epoch, max_items=1):
    model.eval()
    for i, (ev, dep, mask) in enumerate(loader):
        if i >= max_items:
            break
        ev = ev.to(device)
        dep = dep.to(device)
        mask = mask.to(device)

        pred = model(ev)

        # pick first in batch
        ev0 = ev[0:1]       # (1,1,H,W)
        dep0 = dep[0:1]
        pred0 = pred[0:1]
        mask0 = mask[0:1]

        # normalize for display (depth can be in meters)
        def norm01(x):
            x = x.clone()
            x = x - x.min()
            x = x / (x.max().clamp_min(1e-6))
            return x

        writer.add_image("input/events", norm01(ev0[0]), epoch, dataformats="CHW")
        writer.add_image("gt/depth",     norm01(dep0[0]), epoch, dataformats="CHW")
        writer.add_image("pred/depth",   norm01(pred0[0]), epoch, dataformats="CHW")

        err = torch.abs(pred0 - dep0) * mask0
        writer.add_image("error/abs",    norm01(err[0]), epoch, dataformats="CHW")



# -----------------------------
# Dataset: matches your HDF5 layout
# -----------------------------
@dataclass
class PairingConfig:
    offset: int = 0  # event_index = depth_index + offset
    crop_size: Optional[Tuple[int, int]] = None  # (H, W)
    resize_to_depth: bool = True  # resize event frames to depth resolution
    depth_scale: float = 0.001  # if depth stored in mm -> meters
    clamp_depth: Optional[Tuple[float, float]] = (0.2, 6.0)  # meters, optional
    event_norm: str = "01"  # "01" => /255, "none" => raw float
    event_log1p: bool = False  # if your frames are more count-like, can help


class EventDepthIndexDataset(Dataset):
    """
    Pairs:
      events:  data/hdf5/events_camX.h5 -> group "events/frames" uint8 (M, He, We)
      depth:   data/hdf5/realsense.h5   -> group "realsense/depth" uint16 (N, 480, 640)

    Sync:
      by index with an optional offset:
        event_idx = depth_idx + offset
    """
    def __init__(self, event_h5: str, rs_h5: str, cfg: PairingConfig, split: str = "train", val_ratio: float = 0.1):
        super().__init__()
        self.event_h5_path = event_h5
        self.rs_h5_path = rs_h5
        self.cfg = cfg

        with h5py.File(self.event_h5_path, "r") as h5e:
            self.ev_frames = h5e["events/frames"]
            self.M = self.ev_frames.shape[0]
            self.ev_h = self.ev_frames.shape[1]
            self.ev_w = self.ev_frames.shape[2]

        with h5py.File(self.rs_h5_path, "r") as h5r:
            self.rs_depth = h5r["realsense/depth"]
            self.N = self.rs_depth.shape[0]
            self.d_h = self.rs_depth.shape[1]
            self.d_w = self.rs_depth.shape[2]

        # Determine valid depth indices that have a paired event index
        # event_idx = depth_idx + offset must be in [0, M-1]
        depth_min = 0
        depth_max = self.N - 1

        if cfg.offset >= 0:
            depth_min = 0
            depth_max = min(depth_max, self.M - 1 - cfg.offset)
        else:
            depth_min = max(depth_min, -cfg.offset)
            depth_max = depth_max

        if depth_max < depth_min:
            raise RuntimeError(
                f"No overlap between depth and event frames with offset={cfg.offset}. "
                f"N_depth={self.N}, N_events={self.M}"
            )

        all_depth_indices = np.arange(depth_min, depth_max + 1, dtype=np.int64)

        n_total = len(all_depth_indices)
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_val

        # --- block-based random split ---
        block_size = 10  # choose e.g. 8, 16, 32 depending on how similar neighbors are

        rng = np.random.default_rng()

        # number of full blocks (last block may be shorter)
        n_blocks = (n_total + block_size - 1) // block_size
        block_ids = np.arange(n_blocks, dtype=np.int64)
        rng.shuffle(block_ids)

        val_mask = np.zeros(n_total, dtype=bool)
        val_count = 0

        for b in block_ids:
            start = b * block_size
            end = min((b + 1) * block_size, n_total)
            block_len = end - start

            # add whole blocks to val until we reach/exceed target
            if val_count < n_val:
                val_mask[start:end] = True
                val_count += block_len

        train_indices = all_depth_indices[~val_mask]
        val_indices   = all_depth_indices[val_mask]

        if split == "train":
            self.depth_indices = train_indices
        elif split == "val":
            self.depth_indices = val_indices
        else:
            raise ValueError("split must be 'train' or 'val'")


    def __len__(self):
        return len(self.depth_indices)

    def _center_crop(self, x: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
        h, w = x.shape[-2], x.shape[-1]
        if crop_h > h or crop_w > w:
            raise ValueError(f"Crop {crop_h}x{crop_w} bigger than {h}x{w}")
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        return x[..., y0:y0 + crop_h, x0:x0 + crop_w]

    def __getitem__(self, i: int):
        depth_idx = int(self.depth_indices[i])
        event_idx = depth_idx + self.cfg.offset

        # Load frames (open per call => safe with DataLoader workers)
        with h5py.File(self.event_h5_path, "r") as h5e:
            ev = h5e["events/frames"][event_idx]  # (He,We) uint8

        with h5py.File(self.rs_h5_path, "r") as h5r:
            dep = h5r["realsense/depth"][depth_idx]  # (H,W) uint16

        # --- events to float32 ---
        ev = ev.astype(np.float32)
        if self.cfg.event_norm == "01":
            ev = ev / 255.0

        if self.cfg.event_log1p:
            ev = np.log1p(np.maximum(ev, 0.0))

        # --- depth to meters float32 ---
        dep = dep.astype(np.float32) * float(self.cfg.depth_scale)

        # Valid mask (RealSense: 0 usually means invalid)
        mask = (dep > 0.0).astype(np.float32)

        if self.cfg.clamp_depth is not None:
            dmin, dmax = self.cfg.clamp_depth
            dep = np.clip(dep, dmin, dmax)
            # keep mask as-is; clamping doesn't change validity

        # Resize events to depth resolution if needed
        # (common: event cam resolution differs)
        if self.cfg.resize_to_depth and (ev.shape[0] != dep.shape[0] or ev.shape[1] != dep.shape[1]):
            # Use torch interpolate later? Here with cv2-like via torch for fewer deps
            ev_t = torch.from_numpy(ev)[None, None, ...]  # (1,1,He,We)
            ev_t = F.interpolate(ev_t, size=(dep.shape[0], dep.shape[1]), mode="bilinear", align_corners=False)
            ev = ev_t[0, 0].numpy()

        # Add channel dims
        ev = ev[None, ...]     # (1,H,W)
        dep = dep[None, ...]   # (1,H,W)
        mask = mask[None, ...] # (1,H,W)

        # Optional center crop
        if self.cfg.crop_size is not None:
            ch, cw = self.cfg.crop_size
            ev = self._center_crop(ev, ch, cw)
            dep = self._center_crop(dep, ch, cw)
            mask = self._center_crop(mask, ch, cw)

        # To torch float32 tensors
        ev_t = torch.from_numpy(ev).float()
        dep_t = torch.from_numpy(dep).float()
        mask_t = torch.from_numpy(mask).float()

        return ev_t, dep_t, mask_t


# -----------------------------
# Train / Val loops
# -----------------------------
def train_one_epoch(model, loader, opt, device, lambda_smooth: float):
    model.train()
    running = 0.0

    for ev, dep, mask in loader:
        ev = ev.to(device, non_blocking=True)     # (B,1,H,W)
        dep = dep.to(device, non_blocking=True)   # (B,1,H,W)
        mask = mask.to(device, non_blocking=True) # (B,1,H,W)

        pred = model(ev)

        loss = masked_charbonnier(pred, dep, mask)
        if lambda_smooth > 0:
            ref = ev  # event intensity as edge reference
            loss = loss + lambda_smooth * edge_aware_smoothness(pred, ref, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += float(loss.item())

    return running / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running = 0.0
    for ev, dep, mask in loader:
        ev = ev.to(device, non_blocking=True)
        dep = dep.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(ev)
        loss = masked_charbonnier(pred, dep, mask)
        running += float(loss.item())
    return running / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--events_h5",
        type=str,
        default=str(DEFAULT_EVENTS_CAM0),
        help="Path to event frames HDF5 (default: events_cam0.h5)"
    )

    ap.add_argument(
        "--realsense_h5",
        type=str,
        default=str(DEFAULT_REALSENSE),
        help="Path to RealSense depth HDF5"
    )


    ap.add_argument("--offset", type=int, default=0, help="event_idx = depth_idx + offset (index sync)")
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--depth_scale", type=float, default=0.001, help="uint16 depth * depth_scale => meters")
    ap.add_argument("--clamp_min", type=float, default=0.2)
    ap.add_argument("--clamp_max", type=float, default=6.0)

    ap.add_argument("--no_resize_to_depth", action="store_true")
    ap.add_argument("--crop_h", type=int, default=0)
    ap.add_argument("--crop_w", type=int, default=0)

    ap.add_argument("--event_norm", type=str, default="01", choices=["01", "none"])
    ap.add_argument("--event_log1p", action="store_true")

    ap.add_argument("--lambda_smooth", type=float, default=0.01)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR)
    )


    args = ap.parse_args()

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.out_dir, "runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard logdir:", log_dir)


    crop_size = None
    if args.crop_h > 0 and args.crop_w > 0:
        crop_size = (args.crop_h, args.crop_w)

    cfg = PairingConfig(
        offset=args.offset,
        crop_size=crop_size,
        resize_to_depth=(not args.no_resize_to_depth),
        depth_scale=args.depth_scale,
        clamp_depth=(args.clamp_min, args.clamp_max),
        event_norm=args.event_norm,
        event_log1p=args.event_log1p,
    )

    train_ds = EventDepthIndexDataset(
        args.events_h5, args.realsense_h5, cfg=cfg, split="train", val_ratio=args.val_ratio
    )
    val_ds = EventDepthIndexDataset(
        args.events_h5, args.realsense_h5, cfg=cfg, split="val", val_ratio=args.val_ratio
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.version())

    model = UNet(in_channels=1, base=args.base, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, lambda_smooth=args.lambda_smooth)
        va = validate(model, val_loader, device)

        writer.add_scalar("loss/train", tr, epoch)
        writer.add_scalar("loss/val", va, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        if device.type == "cuda":
            writer.add_scalar("gpu/mem_alloc_mb", torch.cuda.memory_allocated() / 1024**2, epoch)
            writer.add_scalar("gpu/mem_reserved_mb", torch.cuda.memory_reserved() / 1024**2, epoch)



        print(f"Epoch {epoch:03d} | train {tr:.6f} | val {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "val_loss": va,
                    "cfg": cfg.__dict__,
                },
                os.path.join(args.out_dir, "best.pt"),
            )
        if epoch % 1 == 0:
            log_images(writer, model, val_loader, device, epoch)


    print(f"Done. Best val: {best_val:.6f}. Saved: {os.path.join(args.out_dir, 'best.pt')}")

    writer.close()



if __name__ == "__main__":
    main()
