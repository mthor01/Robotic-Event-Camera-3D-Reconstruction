#!/usr/bin/env python3
"""
unet_viz.py — TensorBoard image visualization helper for train_unet.py.

Each logged panel contains up to `n_samples` rows, one row per sample.
Each row has six sub-images (left to right):

  events | gt depth | mask | pred depth | error | overlay (events + pred depth)

Usage in train_unet.py
----------------------
    from unet_viz import UNetVizLogger

    viz_train = UNetVizLogger(writer, n_samples=4, tag="viz/train")
    viz_val   = UNetVizLogger(writer, n_samples=4, tag="viz/val")

    # Inside run_epoch (or the training loop):
    viz_train.add_batch(voxels, depth, mask, pred)

    # At the end of each epoch:
    viz_train.flush(step=epoch)
    viz_val.flush(step=epoch)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------------
# Colourmap helpers
# ---------------------------------------------------------------------------

def _get_cmap(name: str):
    """Return a matplotlib colormap, compatible with old and new matplotlib."""
    try:
        import matplotlib
        return matplotlib.colormaps[name]          # matplotlib >= 3.7
    except AttributeError:
        import matplotlib.cm as cm
        return cm.get_cmap(name)                   # matplotlib < 3.7


_CMAP_TURBO  = None
_CMAP_PLASMA = None
_CMAP_GRAY   = None


def _cmap_turbo():
    global _CMAP_TURBO
    if _CMAP_TURBO is None:
        _CMAP_TURBO = _get_cmap("turbo")
    return _CMAP_TURBO


def _cmap_plasma():
    global _CMAP_PLASMA
    if _CMAP_PLASMA is None:
        _CMAP_PLASMA = _get_cmap("plasma")
    return _CMAP_PLASMA


def _cmap_gray():
    global _CMAP_GRAY
    if _CMAP_GRAY is None:
        _CMAP_GRAY = _get_cmap("gray")
    return _CMAP_GRAY


# ---------------------------------------------------------------------------
# Low-level image building utilities
# ---------------------------------------------------------------------------

_BORDER_PX    = 2          # pixel width of separators
_BORDER_VALUE = 0.45       # separator brightness (mid-gray)
_INVALID_VALUE = 0.10      # color for masked-out pixels

# Depth range used for colormap normalisation
_DEPTH_MIN = 0.05          # metres  (matches config.DEPTH_MIN)
_DEPTH_MAX = 0.60          # metres  (matches config.D_MAX)


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Detach, move to CPU, convert to float32 numpy array."""
    return t.detach().float().cpu().numpy()


def _norm01(arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Linearly map arr to [0, 1]; clamps to that range."""
    lo = float(arr.min()) if vmin is None else vmin
    hi = float(arr.max()) if vmax is None else vmax
    span = hi - lo
    if span < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / span, 0.0, 1.0)


def _colorize(arr01: np.ndarray, cmap) -> np.ndarray:
    """Apply a matplotlib colormap to a (H, W) float32 in [0,1] → (3, H, W)."""
    rgba = cmap(arr01)           # (H, W, 4)
    return rgba[..., :3].astype(np.float32).transpose(2, 0, 1)  # (3, H, W)


def _events_panel(voxels: np.ndarray) -> np.ndarray:
    """
    voxels : (C, H, W) float32
    Returns (3, H, W) grayscale image of total event activity.
    """
    activity = np.abs(voxels).sum(axis=0)          # (H, W)
    return _colorize(_norm01(activity), _cmap_gray())


def _depth_panel(
    depth: np.ndarray,    # (H, W) metres
    mask:  np.ndarray | None = None,    # (H, W) binary float32
) -> np.ndarray:
    """Turbo-coloured depth map; invalid pixels are dark when mask is given."""
    norm = _norm01(depth, vmin=_DEPTH_MIN, vmax=_DEPTH_MAX)
    rgb  = _colorize(norm, _cmap_turbo())           # (3, H, W)
    if mask is not None:
        rgb[:, mask < 0.5] = _INVALID_VALUE
    return rgb


def _mask_panel(mask: np.ndarray) -> np.ndarray:
    """(H, W) binary float32 → (3, H, W) grayscale."""
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    return np.stack([m, m, m], axis=0)


def _error_panel(
    pred:    np.ndarray,  # (H, W) metres
    gt:      np.ndarray,  # (H, W) metres
    mask:    np.ndarray,  # (H, W) binary float32
) -> np.ndarray:
    """Plasma-coloured absolute error; invalid pixels are dark."""
    err  = np.abs(pred - gt) * mask                 # (H, W)
    norm = _norm01(err, vmin=0.0, vmax=max(float(err.max()), 0.01))
    rgb  = _colorize(norm, _cmap_plasma())          # (3, H, W)
    rgb[:, mask < 0.5] = _INVALID_VALUE
    return rgb


def _overlay_panel(
    voxels: np.ndarray,   # (C, H, W)
    pred:   np.ndarray,   # (H, W) metres
    mask:   np.ndarray,   # (H, W) binary float32
    alpha:  float = 0.55,
) -> np.ndarray:
    """
    Events (grayscale background) blended with predicted depth (turbo)
    wherever the spatial mask is valid.
    """
    # Grayscale event background
    activity = np.abs(voxels).sum(axis=0)           # (H, W)
    ev01     = _norm01(activity)
    bg       = np.stack([ev01, ev01, ev01], axis=0) # (3, H, W)

    # Depth colourmap foreground
    depth_rgb = _colorize(_norm01(pred, _DEPTH_MIN, _DEPTH_MAX), _cmap_turbo())

    valid = mask > 0.5                              # (H, W) bool
    out   = bg.copy()
    out[:, valid] = (1.0 - alpha) * bg[:, valid] + alpha * depth_rgb[:, valid]
    return out


def _build_row(
    voxels:   np.ndarray,  # (C, H, W)
    depth_gt: np.ndarray,  # (H, W) metres
    mask:     np.ndarray,  # (H, W) binary
    pred:     np.ndarray,  # (H, W) metres
) -> np.ndarray:
    """
    Build one horizontal strip of six panels for a single sample.
    Returns (3, H, 6*W + 5*BORDER_PX).
    """
    H = voxels.shape[1]
    sep = np.full((3, H, _BORDER_PX), _BORDER_VALUE, dtype=np.float32)

    panels = [
        _events_panel(voxels),
        sep,
        _depth_panel(depth_gt, mask),
        sep,
        _mask_panel(mask),
        sep,
        _depth_panel(pred, None),
        sep,
        _error_panel(pred, depth_gt, mask),
        sep,
        _overlay_panel(voxels, pred, mask),
    ]
    return np.concatenate(panels, axis=2)   # (3, H, W_total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VizLogger:
    """
    Accumulates sample frames during a training/validation epoch and writes
    a visualization panel to TensorBoard.

    Each sample row shows (left → right):
        events | gt depth | mask | pred depth | error | overlay

    Parameters
    ----------
    writer : SummaryWriter
        Active TensorBoard writer.
    n_samples : int
        How many sample rows to show in the panel (default 4).
    tag : str
        TensorBoard tag, e.g. ``"viz/train"`` or ``"viz/val"``.
    """

    def __init__(
        self,
        writer:    SummaryWriter,
        n_samples: int = 4,
        tag:       str = "viz",
    ) -> None:
        self.writer    = writer
        self.n_samples = n_samples
        self.tag       = tag
        self._buf: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._seen = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Discard all accumulated samples."""
        self._buf.clear()
        self._seen = 0

    # ------------------------------------------------------------------
    def add_batch(
        self,
        voxels: torch.Tensor,  # (B, C, H, W)
        depth:  torch.Tensor,  # (B, 1, H, W)  ground-truth depth in metres
        mask:   torch.Tensor,  # (B, 1, H, W)  binary validity mask
        pred:   torch.Tensor,  # (B, 1, H, W)  predicted depth in metres
    ) -> None:
        """
        Store representative frames from this batch using reservoir sampling.

        Instead of always taking the first ``n_samples`` frames in the epoch,
        this gives each seen frame equal probability of ending up in the
        visualization buffer. This greatly improves variety for ordered
        validation loaders.

        Call this once per batch during your epoch loop.
        """
        v_np = _to_np(voxels)          # (B, C, H, W)
        d_np = _to_np(depth[:, 0])     # (B, H, W)
        m_np = _to_np(mask[:, 0])      # (B, H, W)
        p_np = _to_np(pred[:, 0])      # (B, H, W)

        for b in range(v_np.shape[0]):
            sample = (v_np[b], d_np[b], m_np[b], p_np[b])
            self._seen += 1

            # Fill the reservoir first, then randomly replace existing samples.
            if len(self._buf) < self.n_samples:
                self._buf.append(sample)
                continue

            j = int(self._rng.integers(0, self._seen))
            if j < self.n_samples:
                self._buf[j] = sample

    # ------------------------------------------------------------------
    def flush(self, step: int) -> None:
        """
        Build the visualization panel from accumulated samples, write it to
        TensorBoard, and reset the internal buffer.

        Call once at the end of each epoch.
        """
        if not self._buf:
            return

        rows = [_build_row(*s) for s in self._buf]

        # Horizontal separator between sample rows
        W_total = rows[0].shape[2]
        h_sep   = np.full((3, _BORDER_PX, W_total), _BORDER_VALUE, dtype=np.float32)

        parts: List[np.ndarray] = []
        for i, row in enumerate(rows):
            if i > 0:
                parts.append(h_sep)
            parts.append(row)

        panel = np.clip(np.concatenate(parts, axis=1), 0.0, 1.0)  # (3, H_total, W_total)

        self.writer.add_image(self.tag, panel, global_step=step)
        self.writer.flush()   # force write to disk so TensorBoard shows images immediately
        self.reset()
