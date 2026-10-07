# Robot Recording and Event-Based 3-D Reconstruction

This repository contains the code of my master's thesis on event-based
multi-view depth estimation and 3-D reconstruction.

**[Read the master's thesis (PDF)](https://github.com/mthor01/robot_and_record/blob/main/docs/Masters_Thesis_github.pdf)**
· **[Dataset on Hugging Face](https://huggingface.co/datasets/mthor/Event_and_Depth)**

The main contribution is an event-based depth-estimation method inspired by
RGB multi-view stereo (MVS). Instead of matching conventional RGB images, the
model combines event voxel grids from several calibrated viewpoints with an
approximate table-plane prior. It uses camera geometry to construct
coarse-to-fine cost volumes, predicts a dense depth map and confidence map for
each reference view, and fuses the resulting predictions into a 3-D mesh.

## Two versions of this repository

| Branch | Contents | Intended use |
| --- | --- | --- |
| [`main`](https://github.com/mthor01/robot_and_record/tree/main) | The complete thesis project: Franka robot-arm control, trajectory execution, synchronized data recording, camera and hand-eye calibration, the U-Net baselines and model comparisons, the architecture and training variants explored during development, and many analysis, visualization, and test scripts used throughout the thesis. | Documents and reproduces the full experimental system. |
| **`light`** (this branch) | Everything needed to work with the published dataset: preprocessing, training, evaluation, and TSDF reconstruction of the MVS model. | Using, retraining, and extending the method. |

The `main` branch is very setup specific. Its recording and robot-control code
depends on our particular hardware: a Franka Emika Panda with the Deoxys
control stack, a hardware-synchronized Intel RealSense D435 and IDS event
camera on a custom end-effector mount, and our calibration, network
configuration, and vendor drivers. It is not intended as a generic
data-collection system for arbitrary robots or cameras, and much of it only
runs on that setup.

For convenience, we therefore provide this **light** version. It starts from
the recorded dataset, which can be downloaded with a single command, and
contains only the code of the final method, without the hardware-specific
parts and the experimental variants. Its defaults reproduce the configuration
of the model reported in the thesis.

## What you can do with the light version

- **Download the dataset** of 50 recorded sequences, either with precomputed
  model inputs or as raw recordings ([Dataset](#dataset)).
- **Preprocess raw recordings** into event voxel grids, ground-truth depth in
  the event-camera frame, and table-plane priors, for the published raw files
  or your own recordings in the same format
  ([Preprocessing](#preprocessing-optional)).
- **Train the MVS model** with the thesis configuration or with your own
  hyperparameters, and monitor training in TensorBoard
  ([Training](#training)).
- **Evaluate depth predictions** with standard depth metrics for the whole
  frame and for spatial regions around the object ([Evaluation](#evaluation)).
- **Reconstruct 3-D meshes** by fusing predicted depth maps with uniform or
  confidence-weighted TSDF integration, and compare them with meshes from the
  ground-truth depth ([Reconstruction](#reconstruction)).

## Method overview

An Intel RealSense D435 RGB-D camera and an IDS UE-39B0XCP-E event camera are
rigidly attached to the end effector of a Franka Emika Panda robot. The cameras
are hardware synchronized, while the robot publishes timestamped end-effector
poses. Calibration provides the transformations needed both to project
RealSense depth into the event-camera frame and to place all event-camera
observations in a common world coordinate system, the robot base frame. The
RealSense depth serves only as supervision; the model sees only events and the
table prior.

1. **Recording** (`main` branch). The robot moves the cameras along a smooth
   multi-view trajectory around a static object.
2. **Preprocessing.** RealSense depth is projected into the event-camera
   frame, events are accumulated into five-bin voxel grids per frame, and a
   view-dependent table-plane depth prior is generated.
3. **MVS-inspired depth prediction.** One event observation is the reference
   view and up to eight displaced observations of the same scene are source
   views. A shared feature pyramid extracts multi-scale features, calibrated
   camera geometry warps source-view features across depth hypotheses, and
   three cascaded cost volumes progressively narrow the depth interval. A
   refinement network predicts full-resolution depth and a per-pixel
   confidence map.
4. **TSDF reconstruction.** Predicted depth maps are transformed into the
   world frame and integrated into a truncated signed distance field.
   Confidence-weighted fusion reduces the influence of unreliable predictions,
   and the surface is extracted as a triangle mesh.

```text
event streams + RealSense depth + robot poses        (recording, main branch)
                      |
                      v
 event voxel grids + table priors + ground-truth depth       (preprocessing)
                      |
                      v
 geometry-aware multi-view depth and confidence prediction      (MVS model)
                      |
                      v
        confidence-weighted TSDF fusion -> 3-D mesh          (reconstruction)
```

## Results

The results are measured on the six held-out evaluation sequences, whose
objects were not used for training, inside the workspace cube around the
object. The model achieves a mean absolute depth error (MAE) of **0.180 cm**
and a root mean squared error (RMSE) of **0.630 cm**.

For reconstruction, the predicted depth maps are fused with confidence-weighted
TSDF integration and compared with reference meshes fused from the
ground-truth depth maps. Accuracy measures predicted-to-reference surface
distance, completeness measures reference-to-predicted distance, and their
symmetric mean is the Chamfer distance; lower is better. Normal consistency is
better when closer to one. Averaged over the six evaluation objects, the
reconstructed meshes achieve an accuracy of **0.108 cm**, completeness of
**0.108 cm**, Chamfer distance of **0.108 cm**, and normal consistency of
**0.966**.

## Repository layout

| Path | Purpose |
| --- | --- |
| `download_dataset.py` | Downloads the dataset from the Hugging Face Hub into `data/`. |
| `config.py` | Shared constants: dataset location, input resolution, depth range, workspace cube, and TSDF settings. |
| `helpers.py` | Camera geometry, dataset discovery, workspace masks, and source-view selection. |
| `camera_data/` | Calibration of the recording setup (intrinsics, extrinsics, depth scale). |
| `data_precomputation/` | Preprocessing of raw recordings into model inputs. |
| `training/train_mvs.py`, `train_mvs.sh` | MVS network, dataset loader, and training loop, with its launcher. |
| `training/evaluation.py`, `evaluation.sh` | Depth evaluation, with its launcher. |
| `training/depth_losses.py`, `tensorboard_helper.py` | Training losses and TensorBoard logging. |
| `reconstruction.py`, `reconstruction.sh` | TSDF reconstruction and surface metrics, with its launcher. |
| `docker_installation/training_and_reconstruction/` | Dockerfile of the software environment. |
| `data/` | Datasets (created by `download_dataset.py`, not tracked by git). |

## Installation

Everything runs inside the Docker image defined in
`docker_installation/training_and_reconstruction/`. It contains PyTorch,
Open3D, the Metavision SDK (needed only to read raw event files during
preprocessing), and `huggingface_hub`. A Linux machine with an NVIDIA GPU and
the NVIDIA Container Toolkit is recommended for training and reconstruction.

Build the image from the repository root. Supplying the local user and group
IDs keeps files written into the mounted repository owned by you:

```bash
docker build \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  -t robot-record-reconstruction \
  docker_installation/training_and_reconstruction
```

Start a container with the repository mounted at `/workspace`. The port
mapping is only needed for TensorBoard:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p 6006:6006 \
  -v "$(pwd):/workspace" \
  -w /workspace \
  robot-record-reconstruction bash
```

All commands below are run from this container shell in `/workspace`. Adjust
`--shm-size`, batch sizes, and worker counts to the available memory and GPU.

## Dataset

### Download

The dataset is published on the Hugging Face Hub as
[`mthor/Event_and_Depth`](https://huggingface.co/datasets/mthor/Event_and_Depth).
Download it into `data/Event_and_Depth/`, the default dataset location of all
scripts and launchers, with:

```bash
python3 download_dataset.py
```

By default this downloads every file of the 42 training and 6 evaluation
sequences (about 154 GB). Training, evaluation, and reconstruction only need
the precomputed model inputs, which `--precomputed_only` selects (about
60 GB). Options:

| Option | Effect |
| --- | --- |
| `--precomputed_only` | Voxel grids, ground-truth depth, poses, and table priors: the inputs of training, evaluation, and reconstruction (about 60 GB). |
| `--raw_only` | Raw event streams, RealSense recordings, and poses: the inputs of the preprocessing (about 91 GB). |
| `--eval_only` | Only the evaluation sequences (about 21 GB). Can be combined with `--precomputed_only` (about 8 GB) or `--raw_only` (about 12 GB). |
| `--sequences 20 25` | Download only the named sequences. |
| `--dry_run` | Print the number of files and the download size without downloading. |

The script checks the free disk space before downloading. Complete files are
skipped, so an interrupted download can simply be restarted.

### Recordings

The dataset contains 50 object-specific recordings of static, colored
building-block structures on a mostly textureless white table. For every
recording, the robot moves the camera rig along a smooth path through randomly
sampled, reachable viewpoints on a restricted hemisphere around the object.
Each trajectory is constructed from 30 target poses and is recorded
continuously, producing synchronized event streams, RGB-D measurements, and
camera poses at 30 frames per second.

The recordings are organized into four folders. Each recording is one
**sequence directory**, named by its recording number:

| Folder | Sequences | Frames per sequence | Use |
| --- | --- | --- | --- |
| `train/` | 42: 21–24, 26–29, 31–39, 41–44, 46–49, 51–59, 61–64, 66–69 | 798–1571 (43,543 in total) | Training |
| `eval/` | 6: 20, 25, 30, 50, 65, 70 | 922–1590 (6,959 in total) | Validation during training and the evaluation reported in the thesis |
| `special/` | 2: 40, 60 | 989 and 999 | Two additional recordings outside the training/evaluation split |
| `eval_and_special/` | 8: the sequences of `eval/` and `special/` | | All eight non-training sequences in one folder, for example to evaluate them in one run |

The objects of the evaluation sequences do not appear in the training
sequences, and there is no separate test set. `eval_and_special/` contains the
same data as `eval/` and `special/`, so it does not need to be downloaded in
addition to those two folders. `download_dataset.py` fetches only `train/` and
`eval/`; `special/` and `eval_and_special/` can be downloaded from the dataset
page on the Hugging Face Hub.

### Sequence directory

Every sequence directory contains the raw recording and the precomputed model
inputs derived from it:

```text
<split>/<sequence>/
├── raw_event_data/
│   └── events_cam0.raw          raw    event stream with hardware triggers
├── events/
│   └── voxels_cam0.h5           model  event voxel grid per frame
└── hdf5/
    ├── realsense.h5             raw    RealSense depth and RGB frames
    ├── poses.h5                 both   robot end-effector pose per frame
    ├── depth_in_event_frame.h5  model  ground-truth depth in the event-camera frame
    ├── table_plane.h5           model  table-plane depth prior
    └── rgb_in_event_frame.h5    other  RGB in the event-camera frame (visualization only)
```

Files marked *model* are the precomputed model inputs (`--precomputed_only`),
files marked *raw* are the inputs of the preprocessing (`--raw_only`), and
`poses.h5` belongs to both. A sequence can be used for training, evaluation,
and reconstruction once its four model inputs exist.

**All frame-indexed arrays of a sequence share the same frame index:** index
`i` refers to the same instant and camera pose in every file, and each file
holds the same number of frames `N` (for example 1,257 for sequence 20).
Consecutive frames are 1/30 s apart.

All image-like arrays derived from the event camera share one image geometry:
the native 1280 × 720 event-camera image is center-cropped to 960 × 720 and
then resized to **320 × 240** (width × height). The camera intrinsics are
transformed in the same way (`helpers.transform_intrinsics`), and each of these
files records the transform in its `intrinsics_transform` attribute
(`"center_crop_resize"`).

### File contents

#### `events/voxels_cam0.h5`: event voxel grids

| Dataset | Shape and type | Contents |
| --- | --- | --- |
| `voxels` | `(N, 5, 240, 320)`, `float16` | Event voxel grid of every frame. |
| `hw_trigger_times_us` | `(N,)`, `int64` | Hardware-trigger timestamp of every frame in event-camera microseconds. |

The events of frame `i` are taken from a window of one frame period
(about 33 ms) centered on trigger `i`, so that the middle bin is aligned with
the RealSense frame. Each event contributes its polarity (+1 or −1) to the two
nearest of the five temporal bins (bilinear in time). The non-zero entries of
each grid are then standardized to zero mean and unit variance; empty voxels
stay zero. The attributes of `voxels` store the native, crop, and output
resolutions (`native_h`/`native_w`, `crop_h`/`crop_w`, `resize_h`/`resize_w`),
`normalized`, and `intrinsics_transform`.

#### `hdf5/depth_in_event_frame.h5`: ground-truth depth

| Dataset | Shape and type | Contents |
| --- | --- | --- |
| `depth` | `(N, 240, 320)`, `float32`, gzip | Depth along the event camera's optical axis in metres. `0` marks pixels without a measurement. |

The RealSense depth is unprojected, transformed into the event-camera frame,
projected with the event-camera intrinsics and distortion, and scattered onto
the event image grid, with corrections for depth bleeding at edges and for
small holes. Values cover the whole visible scene, beyond 1 m; training and
evaluation use the range 0.05–0.7 m (`DEPTH_MIN` and `D_MAX` in `config.py`).

#### `hdf5/table_plane.h5`: table-plane prior

| Dataset | Shape and type | Contents |
| --- | --- | --- |
| `table_plane` | `(N, 240, 320)`, `float32`, gzip | Depth of the horizontal table plane along every pixel ray, normalized as `(depth − 0.05) / (0.7 − 0.05)` and clipped to [0, 1]. |

The plane lies at `z = −0.02 m` in the robot base frame (attribute
`table_z_m`); the file also stores `depth_min`, `depth_max`, and
`intrinsics_transform`. The prior depends only on the camera pose and is the
second model input besides the events.

#### `hdf5/poses.h5`: camera poses

| Dataset | Shape and type | Contents |
| --- | --- | --- |
| `ee_T` | `(N, 4, 4)`, `float64` | Homogeneous end-effector pose `T_base_from_ee` of every frame, in metres, in the robot base frame. |
| `nearest_offset_ms` | `(N,)`, `float64` | Time difference to the robot pose sample assigned to the frame (diagnostic). |

The robot base frame serves as the world frame. The event-camera pose of frame
`i` is `T_base_from_event = ee_T[i] @ inv(T_event_from_rgb @ T_rgb_from_ee)`,
using the hand-eye calibration in `camera_data/`
(`helpers.load_event_calibration`).

#### `hdf5/realsense.h5`: RealSense recording

| Dataset | Shape and type | Contents |
| --- | --- | --- |
| `depth` | `(N, 480, 640)`, `uint16` | Raw depth; multiply by the scale in `camera_data/depth_scale.npz` to obtain metres. |
| `rgb` | `(N, 480, 640, 3)`, `uint8` | Color frames in BGR channel order. |
| `t_global_ms` | `(N,)`, `float64` | RealSense global timestamps in milliseconds. |
| `t_rgb_ms` | `(N,)`, `float64` | Color-frame timestamps in milliseconds. |
| `frame_number` | `(N,)`, `int64` | RealSense frame numbers. |

#### `hdf5/rgb_in_event_frame.h5`: RGB in the event-camera frame

| Dataset | Shape and type | Contents |
| --- | --- | --- |
| `rgb` | `(N, 240, 320, 3)`, `uint8`, gzip | RealSense color projected into the event-camera frame together with the depth; pixels without a sample are black. |

The light version does not use this file; it is useful for visualization.

#### `raw_event_data/events_cam0.raw`: raw events

The raw event stream of the IDS UE-39B0XCP-E camera (Sony IMX636 sensor,
1280 × 720) in Prophesee EVT 3.0 format, readable with the Metavision SDK.
Besides the events (pixel coordinates, polarity, and microsecond timestamps),
it contains one rising-edge external-trigger event per RealSense frame, which
defines the frame alignment of the voxel grids. Some sequences also include a
`.tmp_index` file, a cache that the Metavision SDK regenerates automatically;
the download script skips these.

### Calibration

`camera_data/` contains the calibration of the recording setup, which applies
to every sequence of the dataset:

| File | Contents |
| --- | --- |
| `event_intrinsics.npz` | Event camera: `camera_matrix` (3 × 3), `dist_coeffs`, and `image_size` `[W, H]` = [1280, 720] |
| `rs_depth_intrinsics.npz`, `rs_rgb_intrinsics.npz` | RealSense depth and color cameras: `camera_matrix` and `image_size` |
| `depth_scale.npz` | `scale`: metres per RealSense depth unit |
| `T_event_from_depth.npz`, `T_color_from_depth.npz`, `T_event_from_rgb.npz` | Camera-to-camera transforms `T` (4 × 4) |
| `T_rgb_from_ee.npz` | Hand-eye calibration `T` (4 × 4) of the RealSense color camera |

The remaining files are intermediate results of the calibration procedure on
the `main` branch.

## Usage

The three launchers `training/train_mvs.sh`, `training/evaluation.sh`, and
`reconstruction.sh` collect their settings in a configuration block at the top
of the file. They work with the downloaded dataset as they are; edit the block
to change paths, checkpoints, or parameters. Arguments given on the command
line are appended, which is convenient for short experiments:

```bash
./training/train_mvs.sh --epochs 5 --name smoke_test
```

The scripts find the repository from their own location, so they can be
started from any directory. All outputs are written into the mounted
repository and remain available after the container exits.

### Preprocessing (optional)

The downloaded dataset already contains the model inputs, so this step is only
needed for raw data: after `download_dataset.py --raw_only`, or for your own
recordings in the format described above. It requires `raw_event_data/`,
`hdf5/realsense.h5`, and `hdf5/poses.h5` in every sequence and the
calibration in `camera_data/`.

```bash
# every sequence below data/Event_and_Depth (default)
./data_precomputation/precompute_all.sh

# selected sequences
./data_precomputation/precompute_all.sh \
  --data_dir data/Event_and_Depth/train/21 data/Event_and_Depth/eval/20
```

The script runs depth/RGB projection (`project_realsense_to_event.py`),
table-plane generation (`precompute_table_plane.py`), and voxel generation
(`precompute_voxels.py`) in this order and writes their outputs into each
sequence directory, overwriting existing ones. Stage-specific options can be
passed after the markers `--project`, `--table`, and `--voxel`, for example
`./data_precomputation/precompute_all.sh --voxel --float16`; run each script
with `--help` for its settings.

To use your own dataset, arrange the sequence directories into `train/` and
`eval/` folders as in the published dataset and set the `DATA_DIR` values of
the three launchers accordingly. Recordings from another setup also need their
own calibration in `camera_data/`.

### Training

```bash
./training/train_mvs.sh
```

Training uses the sequences in `data/Event_and_Depth/train/` and validates on
`data/Event_and_Depth/eval/` after every epoch. For each target frame, the
source views are selected by camera motion: up to four earlier and four later
frames, each at least 5 cm away from the previously selected view.

The defaults of `training/train_mvs.py` reproduce the thesis configuration,
and `train_mvs.sh` lists the main settings explicitly so they are easy to
change:

| Group | Options |
| --- | --- |
| Network | `--feature_channels`, `--cost_channels`, `--reference_channels`, `--coarse/middle/fine_hourglass_levels`, `--refiner_channels`, `--refiner_max_residual_m` |
| Depth hypotheses | `--coarse_depths`, `--middle_depths`, `--middle_window`, `--fine_depths`, `--fine_window_min`, `--fine_window_max` |
| Source views | `--num_views`, `--pose_move_threshold`, `--allow_fewer_pose_views` / `--strict_balanced_pose_views` |
| Loss | `--lambda_grad`, `--lambda_normal`, `--uncertainty` / `--no-uncertainty`, `--lambda_confidence`, `--confidence_abs_tolerance`, `--confidence_rel_tolerance` |
| Optimization | `--epochs`, `--batch_size`, `--lr`, `--min_lr`, `--weight_decay`, `--ema_decay` (AdamW with a cosine schedule) |
| Regularization | `--fpn_dropout`, `--reference_dropout`, `--hourglass_dropout`, `--drop_path_rate`, and the `--no_*` switches of the four data augmentations |

Run `python3 training/train_mvs.py --help` for descriptions and defaults. After
every epoch, the script saves the checkpoints with the best validation L1,
p95, and worst-10 % L1 error to `training/checkpoints/mvs/` as
`best_l1_<name>.pth`, `best_p95_<name>.pth`, and `best_l1_worst10_<name>.pth`,
and at the end the final state as `last_<name>.pth`. Checkpoints store all
settings, so evaluation and reconstruction rebuild the network from them
automatically.

Training writes TensorBoard logs to `training/checkpoints/tensorboard/`. With
the container started as shown above, run

```bash
tensorboard --logdir training/checkpoints/tensorboard --host 0.0.0.0 --port 6006
```

and open <http://localhost:6006> to follow the losses and errors, sample
predictions (`viz/`), and the relation between predicted confidence and error
(`uncertainty/`).

### Evaluation

Set `CHECKPOINT` in `training/evaluation.sh` to a trained checkpoint and run:

```bash
./training/evaluation.sh
```

Evaluation predicts depth for the evaluation sequences and compares it with
the ground truth in three regions:

- **Whole frame:** all pixels with valid ground-truth depth.
- **Workspace cube:** pixels whose ground-truth point lies inside a 32 cm cube
  around the object, which contains the object and the surrounding table.
- **Raised cube:** the same cube, raised so that its bottom lies at
  z = 1.5 cm in the robot base frame, above the table, so that it contains
  only the object.

For each region it reports AbsRel, SqRel, MAE, RMSE, RMSE log, and the
δ < 1.25, 1.25², and 1.25³ accuracies. The results are written to
`training/evaluation_results/`:

| File | Contents |
| --- | --- |
| `summary.txt`, `summary.json` | Metrics over all sequences and the evaluation settings |
| `depth_metrics_by_region.csv` | Metrics over all sequences, one row per region |
| `per_sequence_metrics.csv`, `per_frame_metrics.csv` | Metrics of every sequence and every frame |
| `qualitative_depth_results.png` | Events, ground truth, prediction, and error of one random frame per sequence |
| `selected_frames_overview.png` | The same for the frames chosen with `EXAMPLE_SEQUENCES` and `EXAMPLE_FRAMES` in `evaluation.sh` |

A new run overwrites these files. `--fast_mode N` evaluates only every N-th
frame for quick checks.

### Reconstruction

Set `CHECKPOINT` in `reconstruction.sh` and run:

```bash
./reconstruction.sh
```

For every evaluation sequence, the script predicts depth for evenly spaced
frames (`--mesh_frame_count`, 200 in `reconstruction.sh`) and fuses the
predicted and the ground-truth depth maps into TSDF meshes cropped to the
workspace cube. With `--compare_uncertainty_tsdf`, as in `reconstruction.sh`,
it creates two predicted meshes, one with uniform and one with
confidence-weighted fusion, and compares both with the ground-truth mesh:
accuracy, completeness, Chamfer distance, normal consistency, and
precision/recall/F-score at 1, 2, and 5 cm. `--save_largest_connected_surface`
keeps only the largest connected surface of each mesh before computing the
metrics.

The results are written to
`data/Event_and_Depth/eval/reconstruction_output/<checkpoint name>/`:

| File | Contents |
| --- | --- |
| `<sequence>/<sequence>_gt_mesh.obj` | Mesh fused from the ground-truth depth |
| `<sequence>/<sequence>_uniform_mesh.obj`, `..._uncertainty_weighted_mesh.obj` | Meshes fused from the predicted depth |
| `<sequence>/reconstruction_metrics.json`, `.txt` | Surface metrics of the sequence |
| `reconstruction_summary.json`, `.txt` | Means over all sequences and the difference between the two fusion variants |
| `chamfer_by_object.png` | Chamfer distance of every sequence |

The meshes can be inspected with any mesh viewer, for example MeshLab.
