"""PoseUNet v2: multi-view U-Net depth estimation with feature-space warping.

Input to the network
---------------------
    target  : (B, C, H, W)           target event voxels
    sources : (B, N_src, C, H, W)    source event voxels  [optional at inference]
    T_s_from_t : (B, N_src, 4, 4)    T_source_from_target  [optional]

Two-stage architecture
----------------------
Stage 1 – Initial single-frame depth (target only):
    target_voxels
        → shared encoder (head → EncoderBlocks → bottleneck ResBlocks)
        → initial decoder (DecoderBlocks + sigmoid)
        → init_depth  (B, 1, H, W)

Stage 2 – Multi-view feature fusion + refinement:
    For each source frame i:
        source_voxels_i
            → shared encoder  (same weights)
            → src_bottleneck_i  (B, C_bot, H/S, W/S)
        Warp src_bottleneck_i into target view:
            _warp_features(init_depth_metric_downscaled, src_bot_i, T_i, K/S)
            → warped_bot_i  (B, C_bot, H/S, W/S)
    Aggregate: mean_valid(warped_bot_1 … warped_bot_N) → agg_bot
    Fuse:  concat(target_bot, agg_bot)
           → 1×1 Conv + BN + ReLU
           → fused_bot  (B, C_bot, H/S, W/S)
    Refined decoder (fused_bot + target skip connections)
        → final_depth  (B, 1, H, W)

S = 2 ** num_encoders  (e.g. 8 for the default 3-encoder UNet)

Training losses
---------------
    GT Charbonnier on final_depth         (lambda_gt,     default 0.5)
    GT Charbonnier on init_depth (aux)    (lambda_aux,    default 0.1)
    Edge-aware smoothness on final_depth  (lambda_smooth, default 0.01)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import EncoderBlock, DecoderBlock, ResidualBlock
from .e2depth import charbonnier_loss, linear_normalized_to_depth


# ─────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────

def _edge_aware_smoothness(
    inv_depth: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Edge-aware smoothness on mean-normalised inverse depth.

    Parameters
    ----------
    inv_depth : (B, 1, H, W)
    events    : (B, C, H, W)  target event voxels (for edge weights)
    """
    mean_d = inv_depth.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
    d   = inv_depth / mean_d
    img = events.abs().sum(dim=1, keepdim=True)

    dx_d   = torch.abs(d  [:, :, :, :-1] - d  [:, :, :, 1:])
    dy_d   = torch.abs(d  [:, :, :-1, :] - d  [:, :, 1:, :])
    dx_img = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    dy_img = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

    return (dx_d * torch.exp(-dx_img)).mean() + (dy_d * torch.exp(-dy_img)).mean()


def _warp_features(
    depth_m:    torch.Tensor,
    src:        torch.Tensor,
    T_s_from_t: torch.Tensor,
    K:          torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp a source feature map into the target view using per-pixel metric depth.

    All geometric computations run in float32 even inside autocast.

    Parameters
    ----------
    depth_m    : (B, 1, H, W)  metric depth at the feature-map resolution
    src        : (B, C, H, W)  source feature map (any dtype)
    T_s_from_t : (B, 4, 4)    pose mapping target 3-D points → source camera
    K          : (3, 3)        camera intrinsics at the feature-map resolution

    Returns
    -------
    warped : (B, C, H, W)  source features sampled at reprojected coordinates
    valid  : (B, 1, H, W)  float mask (1 = valid projection, same dtype as src)
    """
    B, _, H, W = depth_m.shape
    device = depth_m.device

    df = depth_m.float()
    Tf = T_s_from_t.float()
    Kf = K.float().to(device)

    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    ones  = torch.ones(H * W, dtype=torch.float32, device=device)
    p     = torch.stack([xs.flatten(), ys.flatten(), ones], dim=0)   # (3, HW)

    K_inv = torch.linalg.inv(Kf)
    p_cam = K_inv @ p                                                 # (3, HW)
    d_f   = df.reshape(B, 1, H * W)
    X_t   = p_cam.unsqueeze(0) * d_f                                 # (B, 3, HW)
    X_t_h = torch.cat([X_t, ones.reshape(1, 1, H * W).expand(B, -1, -1)], dim=1)

    X_s_h = torch.bmm(Tf, X_t_h)                                     # (B, 4, HW)
    X_s   = X_s_h[:, :3]

    Kf_b  = Kf.unsqueeze(0).expand(B, -1, -1)
    p_s_h = torch.bmm(Kf_b, X_s)                                     # (B, 3, HW)
    z_s   = p_s_h[:, 2:3].clamp(min=1e-4)
    p_s   = p_s_h[:, :2] / z_s                                       # (B, 2, HW)

    valid_z = p_s_h[:, 2] > 1e-3
    valid_x = (p_s[:, 0] >= 0) & (p_s[:, 0] <= W - 1)
    valid_y = (p_s[:, 1] >= 0) & (p_s[:, 1] <= H - 1)
    valid   = (valid_z & valid_x & valid_y).reshape(B, 1, H, W).to(dtype=src.dtype)

    grid_x = (p_s[:, 0] / (W - 1)) * 2 - 1
    grid_y = (p_s[:, 1] / (H - 1)) * 2 - 1
    grid   = torch.stack([grid_x, grid_y], dim=-1).reshape(B, H, W, 2)

    warped = F.grid_sample(
        src.float(), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).to(dtype=src.dtype)

    return warped, valid


# ─────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────

class PoseUNet(nn.Module):
    """Multi-view depth UNet with bottleneck feature warping.

    Parameters
    ----------
    in_channels   : event voxel bins C
    n_frames      : total frames = 1 target + (n_frames-1) sources
    base          : base filter count (doubled each encoder level)
    num_encoders  : encoder/decoder depth  (bottleneck stride = 2**num_encoders)
    num_residuals : bottleneck residual blocks
    K             : (3,3) intrinsics at training resolution
    input_hw      : (H, W) spatial size of network input
    lambda_gt     : GT Charbonnier weight for refined (final) depth
    lambda_aux    : GT Charbonnier weight for initial (single-frame) depth
    lambda_smooth : edge-aware smoothness weight on final depth
    depth_min/max : depth range in metres
    """

    use_pose_warp: bool = False

    def __init__(
        self,
        in_channels:   int,
        n_frames:      int   = 5,
        base:          int   = 32,
        num_encoders:  int   = 3,
        num_residuals: int   = 2,
        K:             Optional[np.ndarray] = None,
        input_hw:      Optional[Tuple[int, int]] = None,
        lambda_gt:     float = 0.5,
        lambda_aux:    float = 0.1,
        lambda_smooth: float = 0.01,
        depth_min:     float = 0.05,
        depth_max:     float = 3.0,
    ):
        super().__init__()
        self.n_frames      = n_frames
        self.n_src         = n_frames - 1
        self.num_encoders  = num_encoders
        self.depth_min     = depth_min
        self.depth_max     = depth_max
        self.lambda_gt     = lambda_gt
        self.lambda_aux    = lambda_aux
        self.lambda_smooth = lambda_smooth

        K_t = torch.from_numpy(K).float() if isinstance(K, np.ndarray) else (
              K.float() if K is not None else torch.eye(3))
        self.register_buffer("K", K_t)

        # Precompute K at bottleneck resolution for feature warping
        stride  = 2 ** num_encoders
        K_bot   = K_t.clone()
        K_bot[0, 0] /= stride   # fx
        K_bot[0, 2] /= stride   # cx
        K_bot[1, 1] /= stride   # fy
        K_bot[1, 2] /= stride   # cy
        self.register_buffer("K_bot", K_bot)

        # ── Shared encoder ────────────────────────────────────────────
        # Processes both target and source frames with the same weights.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.encoders = nn.ModuleList()
        ch = base
        # skip_chs[i] = channels of the i-th skip connection fed into the decoders
        # skips are: [head_output, enc0_output, enc1_output, ...]  (all except last)
        skip_chs: List[int] = [base]
        for i in range(num_encoders):
            out_ch = ch * 2
            self.encoders.append(EncoderBlock(ch, out_ch))
            if i < num_encoders - 1:
                skip_chs.append(out_ch)
            ch = out_ch
        bot_ch = ch   # bottleneck channels: base * 2^num_encoders

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(bot_ch) for _ in range(num_residuals)]
        )

        # ── Initial decoder (target → init_depth) ────────────────────
        init_dec: List[nn.Module] = []
        ch = bot_ch
        for i in range(num_encoders):
            skip_ch = skip_chs[-(i + 1)]
            out_ch  = ch // 2
            init_dec.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch
        self.init_decoder = nn.ModuleList(init_dec)
        self.init_out     = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

        # ── Fusion: concat(tgt_bot, agg_warped_bots) → fused_bot ─────
        self.fusion = nn.Sequential(
            nn.Conv2d(bot_ch * 2, bot_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(bot_ch),
            nn.ReLU(inplace=True),
        )

        # ── Refined decoder (fused_bot + target skips → final_depth) ─
        ref_dec: List[nn.Module] = []
        ch = bot_ch
        for i in range(num_encoders):
            skip_ch = skip_chs[-(i + 1)]
            out_ch  = ch // 2
            ref_dec.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch
        self.refine_decoder = nn.ModuleList(ref_dec)
        self.refine_out     = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

        # Stores init_depth during forward() for auxiliary supervision in compute_loss()
        self._init_depth_cache: Optional[torch.Tensor] = None

    # ── Internal helpers ─────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Run shared encoder. Returns (skip_list, bottleneck_features)."""
        feat  = self.head(x)
        skips = [feat]
        for i, enc in enumerate(self.encoders):
            feat = enc(feat)
            if i < self.num_encoders - 1:
                skips.append(feat)
        bot = self.bottleneck(feat)
        return skips, bot

    def _decode_init(self, skips: List[torch.Tensor], bot: torch.Tensor) -> torch.Tensor:
        feat = bot
        for i, dec in enumerate(self.init_decoder):
            feat = dec(feat, skips[-(i + 1)])
        return self.init_out(feat)

    def _decode_refine(self, skips: List[torch.Tensor], fused_bot: torch.Tensor) -> torch.Tensor:
        feat = fused_bot
        for i, dec in enumerate(self.refine_decoder):
            feat = dec(feat, skips[-(i + 1)])
        return self.refine_out(feat)

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        events:     torch.Tensor,
        src_voxels: Optional[torch.Tensor] = None,   # (B, N_src, C, H, W)
        T_s_from_t: Optional[torch.Tensor] = None,   # (B, N_src, 4, 4)
        states=None,
        T_rel=None,
    ) -> Tuple[torch.Tensor, List]:
        """
        Parameters
        ----------
        events     : (B, C, H, W)          target event voxels
        src_voxels : (B, N_src, C, H, W)   source frames  [None → single-frame mode]
        T_s_from_t : (B, N_src, 4, 4)      T_source_from_target per source

        Returns
        -------
        depth : (B, 1, H, W)  normalised depth in [0, 1]
        []    : empty state list (stateless)
        """
        # ── Stage 1: single-frame initial depth from target ───────────
        tgt_skips, tgt_bot = self._encode(events)
        init_depth = self._decode_init(tgt_skips, tgt_bot)
        self._init_depth_cache = init_depth

        # Single-frame fallback (inference without sources, or n_frames=1)
        if src_voxels is None or T_s_from_t is None:
            return init_depth, []
        B, N_src, C, H, W = src_voxels.shape
        if N_src == 0:
            return init_depth, []

        # ── Stage 2: encode source frames, warp to target at bottleneck ──
        stride  = 2 ** self.num_encoders
        H_b, W_b = H // stride, W // stride

        # Convert init_depth to metric, detach to block second-order gradients
        # through the warp grid, then downsample to bottleneck resolution.
        init_depth_m = linear_normalized_to_depth(
            init_depth.detach(), self.depth_min, self.depth_max
        )
        depth_bot = F.adaptive_avg_pool2d(init_depth_m, (H_b, W_b))   # (B, 1, H_b, W_b)

        warped_bots: List[torch.Tensor] = []
        for i in range(N_src):
            _, src_bot = self._encode(src_voxels[:, i])
            warped, valid = _warp_features(
                depth_bot, src_bot, T_s_from_t[:, i], self.K_bot
            )
            warped_bots.append(warped * valid)   # zero out invalid pixels

        agg_bot = torch.stack(warped_bots, dim=0).mean(dim=0)   # (B, C_bot, H_b, W_b)

        # ── Stage 3: fuse and decode to refined depth ─────────────────
        # Ensure dtype consistency before concatenating
        agg_bot = agg_bot.to(dtype=tgt_bot.dtype)
        fused   = self.fusion(torch.cat([tgt_bot, agg_bot], dim=1))
        final_depth = self._decode_refine(tgt_skips, fused)

        return final_depth, []

    # ── Loss ─────────────────────────────────────────────────────────

    def compute_loss(
        self,
        pred:   torch.Tensor,
        gt:     torch.Tensor,
        mask:   torch.Tensor,
        events: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        pred   : (B, 1, H, W)  final normalised depth (output of forward)
        gt     : (B, 1, H, W)  GT normalised depth
        mask   : (B, 1, H, W)  validity mask
        events : (B, C, H, W)  target event voxels (edge weights for smoothness)
        """
        loss      = pred.new_zeros(())
        charb_val = aux_val = smooth_val = 0.0

        # 1. GT Charbonnier on final depth
        if self.lambda_gt > 0 and gt is not None:
            charb     = charbonnier_loss(pred, gt, mask)
            loss      = loss + self.lambda_gt * charb
            charb_val = charb.item()

        # 2. Auxiliary GT supervision on initial (single-frame) depth
        if (self.lambda_aux > 0
                and gt is not None
                and self._init_depth_cache is not None):
            aux     = charbonnier_loss(self._init_depth_cache, gt, mask)
            loss    = loss + self.lambda_aux * aux
            aux_val = aux.item()

        # 3. Edge-aware smoothness on final depth
        depth_m   = linear_normalized_to_depth(pred, self.depth_min, self.depth_max)
        inv_depth = 1.0 / depth_m.clamp(min=1e-3)
        smooth    = _edge_aware_smoothness(inv_depth, events)
        loss      = loss + self.lambda_smooth * smooth
        smooth_val = smooth.item()

        metrics = {
            "total":  loss.item(),
            "charb":  charb_val,
            "smooth": smooth_val,
            "photo":  aux_val,    # "photo" slot reused to display aux-depth loss
            "grad":   0.0,
            "normal": 0.0,
            "mean":   0.0,
            "mv":     0.0,
        }
        return loss, metrics


# ─────────────────────────────────────────────────────────────────────
#  Registry entry points
# ─────────────────────────────────────────────────────────────────────

def add_pose_unet_args(parser) -> None:
    """Register PoseUNet-specific CLI arguments."""
    parser.add_argument("--base",          type=int,   default=32)
    parser.add_argument("--num_encoders",  type=int,   default=3)
    parser.add_argument("--num_residuals", type=int,   default=2)
    parser.add_argument("--n_frames",      type=int,   default=5,
                        help="Total frames (1 target + n_frames-1 sources)")
    parser.add_argument("--frame_offset",  type=int,   default=15,
                        help="Frame-index gap between consecutive source frames")
    parser.add_argument("--lambda_gt",     type=float, default=0.5,
                        help="GT Charbonnier weight for refined (final) depth")
    parser.add_argument("--lambda_aux",    type=float, default=0.1,
                        help="GT Charbonnier weight for initial (single-frame) depth")
    # --lambda_smooth is registered by train.py's shared train_group


def build_pose_unet(args, in_channels: int, K_input, input_hw) -> PoseUNet:
    """Build PoseUNet from parsed CLI args."""
    return PoseUNet(
        in_channels=in_channels,
        n_frames=args.n_frames,
        base=args.base,
        num_encoders=args.num_encoders,
        num_residuals=args.num_residuals,
        K=K_input,
        input_hw=input_hw,
        lambda_gt=args.lambda_gt,
        lambda_aux=args.lambda_aux,
        lambda_smooth=args.lambda_smooth,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )

