"""
Model registry for real_train.py.

To add a new model:
  1. Create a new file in training/models/ that implements the model.
  2. Register it here: add an entry to MODEL_REGISTRY mapping a name string
     to a factory function with signature:
         build_fn(args, in_channels, K_input, input_hw) -> nn.Module

The returned module must satisfy the interface expected by train_one_epoch,
validate, and log_images:
  - model(events, states, T_rel=None) -> (pred, new_states)
      events : (B, C, H, W) event voxels for one timestep
      states : model-specific recurrent state, None at sequence start
      T_rel  : (B, 4, 4) relative pose, passed only if model.use_pose_warp is True
      pred   : (B, 1, H, W) depth in linear-normalised [0, 1] space
  - model.use_pose_warp  (bool attribute)
  - model.depth_min, model.depth_max  (float attributes, metres)
  - model.compute_loss(pred, gt, mask, events, pred_prev, T_curr_from_prev,
                       mask_prev, K) -> (loss, metrics_dict)
      All loss hyperparameters are owned by the model and set at build time.
      metrics_dict must contain keys: charb, grad, smooth, normal, mean, mv, total.
"""

from .e2depth import E2DepthNet, build_e2depth, add_e2depth_args
from .mvsnet import MVSNet, build_mvsnet, add_mvsnet_args
from .unet import UNet, build_unet, add_unet_args
from .magnet import MaGNet, build_magnet, add_magnet_args
from .pose_unet import PoseUNet, build_pose_unet, add_pose_unet_args

MODEL_REGISTRY = {
    "e2depth":   build_e2depth,
    "mvsnet":    build_mvsnet,
    "unet":      build_unet,
    "magnet":    build_magnet,
    "pose_unet": build_pose_unet,
}

MODEL_ARG_REGISTRY = {
    "e2depth":   add_e2depth_args,
    "mvsnet":    add_mvsnet_args,
    "unet":      add_unet_args,
    "magnet":    add_magnet_args,
    "pose_unet": add_pose_unet_args,
}

# "sequence" : recurrent single-view (RealDataset, temporal sequences)
# "mvs"      : multi-view stereo (RealMVSDataset, reference + source views)
# "magnet"   : MaGNet (RealMVSDataset, custom NLL training loop)
# "pose"     : pose-supervised single-frame UNet (RealPoseDataset)
MODEL_DATA_TYPE = {
    "e2depth":   "sequence",
    "mvsnet":    "mvs",
    "unet":      "single",   # stateless — no sequences needed
    "magnet":    "magnet",
    "pose_unet": "pose",
}

__all__ = [
    "MODEL_REGISTRY", "MODEL_ARG_REGISTRY", "MODEL_DATA_TYPE",
    "E2DepthNet", "build_e2depth", "add_e2depth_args",
    "MVSNet", "build_mvsnet", "add_mvsnet_args",
    "UNet", "build_unet", "add_unet_args",
    "MaGNet", "build_magnet", "add_magnet_args",
    "PoseUNet", "build_pose_unet", "add_pose_unet_args",
]
