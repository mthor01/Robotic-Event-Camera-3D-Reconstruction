"""
MaGNet (Event-based): Multi-View Depth Estimation by Fusing Single-View Depth
Probability with Multi-View Geometry, adapted for event camera voxel grids.

Based on:
    "Multi-View Depth Estimation by Fusing Single-View Depth Probability with
     Multi-View Geometry" – Bae, Budvytis & Cipolla, CVPR 2022.
    https://github.com/baegwangbin/MaGNet

Key adaptations vs the paper:
  - Input: event voxels (C temporal bins) instead of RGB frames.
  - No pretrained components: D-Net, F-Net and G-Net are all trained end-to-end.
  - Upsampling: bilinear (validate_mvs handles this; RAFT-style mask can be added later).

Data type: "magnet" (uses the same RealMVSDataset as "mvsnet" but has its own
training loop that computes the NLL loss with geometric weighting over iterations).

Interface (identical to MVSNet so validate_mvs works unchanged):
    depth_est, sigma_est, all_preds = model(imgs, proj_mats, depth_values)
        imgs:         (B, V, C, H, W) – event voxels per view
        proj_mats:    (B, V, 3, 4)    – K @ [R|t] at resize resolution
        depth_values: (B, D)           – not used internally (own sampling)
        depth_est:    (B, H/4, W/4)   – final expected depth in metres
        sigma_est:    (B, H/4, W/4)   – final uncertainty in metres
        all_preds:    list of (mu_i, sigma_i) per iteration (for NLL loss)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBnReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x), inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# D-Net: single-view depth probability from event voxels
# ─────────────────────────────────────────────────────────────────────────────

class EventDNet(nn.Module):
    """
    Lightweight encoder that estimates per-pixel depth probability N(μ, σ²)
    at H/4 × W/4.

    Outputs:
        mu:   (B, 1, H/4, W/4) – expected depth in metres (linear activation)
        sigma:(B, 1, H/4, W/4) – uncertainty in metres   (modified ELU: > 0)
        feat: (B, base*4, H/4, W/4) – bottleneck features (for G-Net / upsampler)
    """

    def __init__(self, in_channels: int, base: int = 32):
        super().__init__()
        b = base
        self.head = _ConvBnReLU(in_channels, b, k=5, p=2)          # H × W
        self.enc1 = nn.Sequential(
            _ConvBnReLU(b,     b * 2, s=2),                         # H/2
            _ConvBnReLU(b * 2, b * 2),
        )
        self.enc2 = nn.Sequential(
            _ConvBnReLU(b * 2, b * 4, s=2),                         # H/4
            _ConvBnReLU(b * 4, b * 4),
        )
        self.res1 = _ResBlock(b * 4)
        self.res2 = _ResBlock(b * 4)
        self.feat_channels = b * 4

        self.mu_head  = nn.Conv2d(b * 4, 1, 1)                      # linear → μ
        self.sig_head = nn.Conv2d(b * 4, 1, 1)                      # → σ via modified ELU

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.head(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.res1(x)
        feat = self.res2(x)                         # (B, b*4, H/4, W/4)

        mu  = self.mu_head(feat)                    # (B, 1, H/4, W/4)
        # Modified ELU: f(x) = ELU(x) + 1 + ε  ensures σ > 0 (Eq. 1 in paper)
        sig = F.elu(self.sig_head(feat)) + 1.0 + 1e-4  # (B, 1, H/4, W/4)
        return mu, sig, feat


# ─────────────────────────────────────────────────────────────────────────────
# F-Net: feature extractor for multi-view matching
# ─────────────────────────────────────────────────────────────────────────────

class EventFNet(nn.Module):
    """
    PSMNet-style feature extractor.  Outputs 32-channel feature maps at H/4 × W/4.
    Same architecture as FeatureNet in mvsnet.py.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnReLU(in_channels, 8,  k=3, s=1, p=1),
            _ConvBnReLU(8,           8,  k=3, s=1, p=1),
            _ConvBnReLU(8,           16, k=3, s=2, p=1),   # H/2
            _ConvBnReLU(16,          16, k=3, s=1, p=1),
            _ConvBnReLU(16,          16, k=3, s=1, p=1),
            _ConvBnReLU(16,          32, k=3, s=2, p=1),   # H/4
            _ConvBnReLU(32,          32, k=3, s=1, p=1),
            nn.Conv2d(32, 32, 3, 1, 1),                     # no BN at output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 32, H/4, W/4)


# ─────────────────────────────────────────────────────────────────────────────
# G-Net: update depth probability distribution from thin cost volume
# ─────────────────────────────────────────────────────────────────────────────

class GNet(nn.Module):
    """
    Lightweight UNet that processes the thin cost volume (B, Ns, h, w) and
    outputs (Δμ/σ, log σ_ratio) at the same resolution.

    The mean update is: μ_new = μ + (Δμ/σ) * σ
    The variance update: σ_new = σ * exp(clamp(σ_ratio, -2, 2))
    """

    def __init__(self, Ns: int):
        super().__init__()
        self.enc0 = nn.Sequential(
            _ConvBnReLU(Ns, 32),
            _ConvBnReLU(32, 32),
        )
        self.enc1 = nn.Sequential(
            _ConvBnReLU(32, 64, s=2),
            _ConvBnReLU(64, 64),
        )
        self.dec1 = nn.Sequential(
            _ConvBnReLU(64 + 32, 32),
        )
        self.out = nn.Conv2d(32, 2, 1)

    def forward(self, cost_vol: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(cost_vol)                                      # (B, 32, h, w)
        e1 = self.enc1(e0)                                            # (B, 64, h/2, w/2)
        e1_up = F.interpolate(e1, size=e0.shape[2:], mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat([e1_up, e0], dim=1))                  # (B, 32, h, w)
        return self.out(d)                                             # (B, 2, h, w)


# ─────────────────────────────────────────────────────────────────────────────
# Probabilistic depth sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_bk(Ns: int, beta: float) -> np.ndarray:
    """
    Precompute b_k sampling offsets for probabilistic depth sampling (Eq. 4).

    d_{u,v,k} = μ_{u,v} + b_k * σ_{u,v}

    b_k is the midpoint of the k-th equal-probability bin in [μ ± β·σ].
    Uses torch.erfinv for the Gaussian quantile function to avoid scipy.
    """
    P_star = math.erf(beta / math.sqrt(2.0))  # probability mass covered by [μ ± βσ]
    bk_vals: List[float] = []
    for k in range(1, Ns + 1):
        lo = float(np.clip((k - 1) / Ns * P_star + (1.0 - P_star) / 2.0, 1e-6, 1 - 1e-6))
        hi = float(np.clip(k       / Ns * P_star + (1.0 - P_star) / 2.0, 1e-6, 1 - 1e-6))
        # Gaussian quantile: ndtri(p) = sqrt(2) · erfinv(2p − 1)
        lo_b = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * lo - 1.0)))
        hi_b = math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * hi - 1.0)))
        bk_vals.append(0.5 * (lo_b + hi_b))
    return np.array(bk_vals, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Depth consistency-weighted matching for one source view
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_matching(
    ref_feat: torch.Tensor,
    src_feat: torch.Tensor,
    src_mu:   torch.Tensor,
    src_sig:  torch.Tensor,
    src_proj: torch.Tensor,
    ref_proj: torch.Tensor,
    candidates: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """
    Compute depth consistency-weighted feature matching scores (Eq. 3 + 5).

    Args:
        ref_feat:   (B, Cf, h, w) reference features
        src_feat:   (B, Cf, h, w) source features
        src_mu:     (B, 1, h, w) source D-Net mean (metres)
        src_sig:    (B, 1, h, w) source D-Net sigma (metres, > 0)
        src_proj:   (B, 3, 4) source projection matrix (K @ [R|t]) at coarse res
        ref_proj:   (B, 3, 4) reference projection matrix at coarse res
        candidates: (B, Ns, h, w) per-pixel depth candidates (metres)
        kappa:      consistency window half-width in sigma units

    Returns:
        cost:  (B, Ns, h, w) weighted matching scores from this source view
    """
    B, Cf, h, w = ref_feat.shape
    Ns = candidates.shape[1]
    device = ref_feat.device
    dtype  = ref_feat.dtype

    # Build 4×4 projection matrices.
    # linalg.inv requires float32; compute the transform in fp32 then cast back.
    I4 = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).contiguous()
    src4 = I4.clone(); src4[:, :3, :] = src_proj.float()
    ref4 = I4.clone(); ref4[:, :3, :] = ref_proj.float()

    # T maps from reference image plane to source image plane:
    #   X_src = T[:3,:3] @ [u_ref, v_ref, 1]^T * d  +  T[:3,3]
    T = (src4 @ torch.linalg.inv(ref4)).to(dtype)   # (B, 4, 4)
    R = T[:, :3, :3]                                 # (B, 3, 3)
    t = T[:, :3, 3:4]                                # (B, 3, 1)

    # Reference pixel grid at coarse resolution
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    ones = torch.ones(h, w, dtype=dtype, device=device)
    xyz  = torch.stack([gx, gy, ones], dim=0).reshape(3, h * w)   # (3, h*w)
    xyz  = xyz.unsqueeze(0).expand(B, -1, -1)                      # (B, 3, h*w)

    R_xyz = R @ xyz    # (B, 3, h*w) – rotated direction vectors

    # Pack source D-Net distribution for joint bilinear sampling
    src_dist = torch.cat([src_mu, src_sig], dim=1)  # (B, 2, h, w)

    cost = torch.zeros(B, Ns, h, w, device=device, dtype=dtype)

    for k in range(Ns):
        d_k = candidates[:, k].reshape(B, 1, h * w)              # (B, 1, h*w)

        # Project to source frame: X_src = R @ xyz * d + t
        X_src = R_xyz * d_k + t                                   # (B, 3, h*w)

        # Depth (Z) in source camera frame
        Z = X_src[:, 2].clamp(min=1e-6)                          # (B, h*w)

        # Source pixel coordinates
        p_xy = X_src[:, :2] / Z.unsqueeze(1)                     # (B, 2, h*w)
        nx = p_xy[:, 0] / ((w - 1) * 0.5) - 1.0                 # normalise to [-1, 1]
        ny = p_xy[:, 1] / ((h - 1) * 0.5) - 1.0
        grid = torch.stack([nx, ny], dim=-1).view(B, h, w, 2)

        # Warp source features
        w_feat = F.grid_sample(
            src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B, Cf, h, w)

        # Feature dot-product matching score (Eq. 3)
        score = (ref_feat * w_feat).sum(dim=1)   # (B, h, w)

        # Sample source D-Net distribution at projected location
        w_dist = F.grid_sample(
            src_dist, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B, 2, h, w)
        mu_s  = w_dist[:, 0]                         # (B, h, w) source mean
        sig_s = w_dist[:, 1].clamp(min=1e-6)         # (B, h, w) source sigma

        # Depth consistency weight: binary mask, 1 if depth candidate is within
        # the κ-sigma confidence interval of the source D-Net prediction (Eq. 5).
        z_src = Z.view(B, h, w)                      # (B, h, w)
        dc_w  = ((z_src - mu_s).abs() < kappa * sig_s).to(dtype)

        cost[:, k] = score * dc_w

    return cost   # (B, Ns, h, w)


# ─────────────────────────────────────────────────────────────────────────────
# MaGNet model
# ─────────────────────────────────────────────────────────────────────────────

class MaGNet(nn.Module):
    """
    MaGNet adapted for event camera voxel grids.

    Pipeline (per forward pass):
      1. D-Net: estimate (μ, σ) at H/4 × W/4 for every view.
      2. F-Net: extract 32-dim features at H/4 × W/4 for every view.
      3. Iterate Niter times:
           a. Probabilistic depth sampling: candidates = μ + b_k·σ  (Eq. 4)
           b. For each source view: depth consistency-weighted feature matching
              (Eq. 3 + 5) → thin cost volume (B, Ns, H/4, W/4)
           c. G-Net: updates (μ, σ) from the cost volume.
      4. Return final μ as the depth estimate.
    """

    mvs_mode:      bool = True
    use_pose_warp: bool = False

    def __init__(
        self,
        in_channels:  int,
        depth_min:    float,
        depth_max:    float,
        num_samples:  int   = 5,
        num_iters:    int   = 3,
        beta:         float = 3.0,
        kappa:        float = 5.0,
        gamma:        float = 0.8,
        dnet_base:    int   = 32,
    ):
        super().__init__()
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.Ns        = num_samples
        self.Niter     = num_iters
        self.kappa     = kappa
        self.gamma     = gamma

        self.dnet = EventDNet(in_channels, base=dnet_base)
        self.fnet = EventFNet(in_channels)
        self.gnet = GNet(num_samples)

        # Precomputed b_k constants – registered as a buffer so they move to the
        # correct device automatically and are saved with checkpoints.
        bk = _compute_bk(num_samples, beta)
        self.register_buffer("bk", torch.from_numpy(bk))  # (Ns,)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        imgs:         torch.Tensor,   # (B, V, C, H, W)
        proj_mats:    torch.Tensor,   # (B, V, 3, 4)
        depth_values: torch.Tensor,   # (B, D)  – not used internally
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, V, C, H, W = imgs.shape

        # Scale projection matrices to the coarse H/4 × W/4 resolution,
        # matching the convention used in MVSNet's homo_warping.
        coarse_proj = proj_mats.clone()
        coarse_proj[:, :, :2, :] = coarse_proj[:, :, :2, :] / 4.0
        ref_proj = coarse_proj[:, 0]   # (B, 3, 4)

        # ── Step 1 & 2: D-Net + F-Net for every view ──────────────────────────
        dnet_outs = [self.dnet(imgs[:, v]) for v in range(V)]  # [(mu,sig,feat), ...]
        fnet_outs = [self.fnet(imgs[:, v]) for v in range(V)]  # [(B,32,h,w), ...]

        # Reference view: starting distribution
        mu  = dnet_outs[0][0]   # (B, 1, h, w)
        sig = dnet_outs[0][1]   # (B, 1, h, w)  always > 0

        ref_feat = fnet_outs[0]

        # Source views (pack into a list for convenience)
        src_views = [
            {
                "mu":   dnet_outs[v][0],
                "sig":  dnet_outs[v][1],
                "feat": fnet_outs[v],
                "proj": coarse_proj[:, v],
            }
            for v in range(1, V)
        ]

        all_preds: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # ── Steps 3a–3c: iterative refinement ─────────────────────────────────
        for _it in range(self.Niter):
            sig_abs = sig.abs().clamp_min(1e-6)

            # (a) Probabilistic depth sampling (Eq. 4)
            # candidates: (B, Ns, h, w)
            candidates = (
                mu + self.bk.view(1, self.Ns, 1, 1) * sig_abs
            ).clamp(self.depth_min, self.depth_max)

            # (b) Depth consistency-weighted multi-view matching
            h, w = mu.shape[2], mu.shape[3]
            cost_vol = torch.zeros(B, self.Ns, h, w, device=mu.device, dtype=mu.dtype)

            for src in src_views:
                cost_vol = cost_vol + _weighted_matching(
                    ref_feat,
                    src["feat"],
                    src["mu"],
                    src["sig"].abs().clamp_min(1e-6),
                    src["proj"],
                    ref_proj,
                    candidates,
                    self.kappa,
                )

            # (c) G-Net update
            update = self.gnet(cost_vol)                          # (B, 2, h, w)
            delta_mu_norm = update[:, 0:1]                        # Δμ / σ
            sig_log_ratio = update[:, 1:2].clamp(-2.0, 2.0)      # log(σ_new / σ)

            mu  = (mu + delta_mu_norm * sig_abs).clamp(self.depth_min, self.depth_max)
            sig = sig_abs * torch.exp(sig_log_ratio)              # σ_new = σ · exp(ratio)

            all_preds.append((mu, sig))

        depth_final = mu[:, 0]    # (B, h, w) in metres
        sigma_final = sig[:, 0]   # (B, h, w)

        return depth_final, sigma_final, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def magnet_nll_loss(
    all_preds: List[Tuple[torch.Tensor, torch.Tensor]],
    depth_gt:  torch.Tensor,   # (B, h, w) ground-truth depth in metres
    mask:      torch.Tensor,   # (B, h, w) binary validity mask
    gamma:     float = 0.8,
) -> torch.Tensor:
    """
    Negative log-likelihood loss with geometric weighting across iterations
    (Eq. 2, summed with weights γ^(N_iter-1-i) so the final iteration has weight 1).

    L = Σ_i  γ^(N_iter-1-i)  ·  mean_{u,v}(  ½ log σ²  +  (μ - d_gt)² / (2σ²)  )
    """
    Niter = len(all_preds)
    denom = mask.sum().clamp_min(1.0)
    total = depth_gt.new_zeros(1)

    for i, (mu, sigma) in enumerate(all_preds):
        w      = gamma ** (Niter - 1 - i)
        mu_f   = mu[:, 0] if mu.dim() == 4 else mu      # (B, h, w)
        sig_f  = sigma[:, 0] if sigma.dim() == 4 else sigma
        sig_sq = sig_f ** 2 + 1e-6
        nll    = 0.5 * torch.log(sig_sq) + 0.5 * (mu_f - depth_gt) ** 2 / sig_sq
        total  = total + w * (nll * mask).sum() / denom

    return total


# ─────────────────────────────────────────────────────────────────────────────
# Registry entry points
# ─────────────────────────────────────────────────────────────────────────────

def add_magnet_args(parser) -> None:
    """Register MaGNet-specific CLI arguments."""
    # Shared with MVS dataset (build_mvs_datasets reads these)
    parser.add_argument("--num_views",          type=int,   default=5,
                        help="Number of views (reference + sources) for MaGNet")
    parser.add_argument("--num_depth",          type=int,   default=192,
                        help="Placeholder depth planes (dataset building; not used in matching)")
    parser.add_argument("--view_interval",      type=int,   default=5,
                        help="Frame-index step between reference and each source view")
    # MaGNet-specific
    parser.add_argument("--magnet_num_samples", type=int,   default=5,
                        help="Number of depth candidates per pixel (Ns)")
    parser.add_argument("--magnet_num_iters",   type=int,   default=3,
                        help="Number of iterative refinement steps (Niter)")
    parser.add_argument("--magnet_beta",        type=float, default=3.0,
                        help="Search range half-width in sigma units (β, Eq. 4)")
    parser.add_argument("--magnet_kappa",       type=float, default=5.0,
                        help="Depth consistency window in sigma units (κ, Eq. 5)")
    parser.add_argument("--magnet_gamma",       type=float, default=0.8,
                        help="Geometric loss decay factor across iterations (γ)")
    parser.add_argument("--magnet_dnet_base",   type=int,   default=32,
                        help="Base channel count for D-Net encoder")


def build_magnet(args, in_channels: int, K_input, input_hw) -> MaGNet:
    return MaGNet(
        in_channels  = in_channels,
        depth_min    = args.depth_min,
        depth_max    = args.depth_max,
        num_samples  = args.magnet_num_samples,
        num_iters    = args.magnet_num_iters,
        beta         = args.magnet_beta,
        kappa        = args.magnet_kappa,
        gamma        = args.magnet_gamma,
        dnet_base    = args.magnet_dnet_base,
    )
