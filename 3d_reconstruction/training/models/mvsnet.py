"""
MVSNet: Multi-View Stereo depth estimation from event voxels.
Based on MVSNet (Yao et al., ECCV 2018).
Used via train.py with --model mvsnet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Building blocks
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
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnReLU(in_channels, 8),
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
        self.conv1 = Conv3dBnReLU(8,  16, s=2)
        self.conv2 = Conv3dBnReLU(16, 16)
        self.conv3 = Conv3dBnReLU(16, 32, s=2)
        self.conv4 = Conv3dBnReLU(32, 32)
        self.conv5 = Conv3dBnReLU(32, 64, s=2)
        self.conv6 = Conv3dBnReLU(64, 64)
        self.deconv7  = nn.ConvTranspose3d(64, 32, 3, 2, 1, output_padding=1)
        self.deconv9  = nn.ConvTranspose3d(32, 16, 3, 2, 1, output_padding=1)
        self.deconv11 = nn.ConvTranspose3d(16,  8, 3, 2, 1, output_padding=1)
        self.prob = nn.Conv3d(8, 1, 3, 1, 1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = F.relu(F.interpolate(self.deconv7(x),  size=conv4.shape[2:], mode="trilinear", align_corners=False) + conv4, inplace=True)
        x = F.relu(F.interpolate(self.deconv9(x),  size=conv2.shape[2:], mode="trilinear", align_corners=False) + conv2, inplace=True)
        x = F.relu(F.interpolate(self.deconv11(x), size=conv0.shape[2:], mode="trilinear", align_corners=False) + conv0, inplace=True)
        return self.prob(x).squeeze(1)


# ----------------------------
# Geometry
# ----------------------------

def homo_warping(src_feat, src_proj, ref_proj, depth_values):
    """
    src_feat:     (B, C, H, W)
    src_proj:     (B, 3, 4)
    ref_proj:     (B, 3, 4)
    depth_values: (B, D)
    """
    B, C, H, W = src_feat.shape
    D = depth_values.shape[1]
    device = src_feat.device

    src_proj_4 = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    ref_proj_4 = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    src_proj_4[:, :3, :] = src_proj
    ref_proj_4[:, :3, :] = ref_proj

    proj  = src_proj_4 @ torch.linalg.inv(ref_proj_4)
    rot   = proj[:, :3, :3]
    trans = proj[:, :3, 3:4]

    y, x = torch.meshgrid(
        torch.arange(0, H, dtype=torch.float32, device=device),
        torch.arange(0, W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    xyz = torch.stack((x, y, torch.ones_like(x)), dim=0).view(3, -1).unsqueeze(0).repeat(B, 1, 1)

    rot_xyz       = rot @ xyz
    rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(B, 1, D, 1)
    proj_xyz      = rot_depth_xyz + trans.view(B, 3, 1, 1)

    proj_xy  = proj_xyz[:, :2] / proj_xyz[:, 2:3].clamp(min=1e-6)
    proj_x_n = proj_xy[:, 0] / ((W - 1) / 2) - 1
    proj_y_n = proj_xy[:, 1] / ((H - 1) / 2) - 1

    grid   = torch.stack((proj_x_n, proj_y_n), dim=-1).view(B, D, H * W, 2)
    warped = F.grid_sample(src_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.view(B, C, D, H, W)


def depth_regression(prob_volume, depth_values):
    return torch.sum(prob_volume * depth_values[:, :, None, None], dim=1)


# ----------------------------
# Model
# ----------------------------

class MVSNet(nn.Module):
    """Multi-View Stereo depth estimation network."""

    # Interface flags expected by train.py
    mvs_mode: bool = True
    use_pose_warp: bool = False

    def __init__(self, in_channels: int, depth_min: float, depth_max: float, num_depth: int):
        super().__init__()
        self.depth_min      = depth_min
        self.depth_max      = depth_max
        self.num_depth      = num_depth
        self.depth_interval = (depth_max - depth_min) / num_depth
        self.feature        = FeatureNet(in_channels)
        self.cost_reg       = CostRegNet()

    def forward(self, imgs, proj_mats, depth_values):
        """
        imgs:         (B, V, C, H, W)
        proj_mats:    (B, V, 3, 4)
        depth_values: (B, D)  depth hypothesis planes in metres
        Returns: (depth, prob_volume, photometric_confidence)
        """
        B, V, C, H, W = imgs.shape
        feats = self.feature(imgs.view(B * V, C, H, W))
        _, Cf, h, w = feats.shape
        feats = feats.view(B, V, Cf, h, w)

        ref_feat  = feats[:, 0]
        src_feats = feats[:, 1:]

        ref_proj  = proj_mats[:, 0].clone()
        src_projs = proj_mats[:, 1:].clone()
        ref_proj[:, :2, :]       /= 4.0
        src_projs[:, :, :2, :]   /= 4.0

        volume_sum    = ref_feat.unsqueeze(2).repeat(1, 1, depth_values.shape[1], 1, 1)
        volume_sq_sum = volume_sum ** 2

        for i in range(V - 1):
            warped = homo_warping(src_feats[:, i], src_projs[:, i], ref_proj, depth_values)
            volume_sum    = volume_sum    + warped
            volume_sq_sum = volume_sq_sum + warped ** 2

        volume_variance = volume_sq_sum / V - (volume_sum / V) ** 2
        cost  = self.cost_reg(volume_variance)
        prob  = F.softmax(cost, dim=1)
        depth = depth_regression(prob, depth_values)
        with torch.no_grad():
            confidence = torch.max(prob, dim=1)[0]
        return depth, prob, confidence


# ----------------------------
# Loss / metrics
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
# Registry entry points
# ----------------------------

def add_mvsnet_args(parser) -> None:
    """Register MVSNet-specific CLI arguments."""
    parser.add_argument("--num_views",     type=int, default=5,
                        help="Number of views (reference + sources) for MVSNet")
    parser.add_argument("--num_depth",     type=int, default=192,
                        help="Number of depth hypothesis planes for MVSNet")
    parser.add_argument("--view_interval", type=int, default=5,
                        help="Frame-index step between reference and each source view")


def build_mvsnet(args, in_channels: int, K_input, input_hw) -> MVSNet:
    return MVSNet(
        in_channels=in_channels,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        num_depth=args.num_depth,
    )
