# Robot Recording and Event-Based 3-D Reconstruction (light)

This is the **light** branch: a concise version of the research pipeline
developed for my master's thesis that contains only what is needed to
preprocess recorded data, train and evaluate the MVS-inspired event-based depth
model, and reconstruct meshes through TSDF fusion.

The [`main` branch](https://github.com/mthor01/robot_and_record/tree/main)
contains the complete project, including synchronized data recording, camera
calibration, Franka robot control, the U-Net baselines and model comparisons,
and the visualization and diagnostic tools.

**[Read the master's thesis (PDF)](https://github.com/mthor01/robot_and_record/blob/main/docs/Masters_Thesis_github.pdf)**

The main contribution is an event-based depth-estimation method inspired by
RGB multi-view stereo (MVS). Instead of matching conventional RGB images, the
model combines event voxel grids from several calibrated viewpoints with an
approximate table-plane prior. It uses camera geometry to construct
coarse-to-fine cost volumes, predicts a dense depth map and confidence map for
each reference view, and fuses the resulting predictions into a 3-D mesh.

## Experimental setup and recorded dataset

An Intel RealSense D435 RGB-D camera and an IDS UE-39B0XCP-E event camera are
rigidly attached to the end effector of a Franka Emika Panda robot. The cameras
are hardware synchronized, while the robot publishes timestamped end-effector
poses. Camera, hand-eye, and robot calibration provide the transformations
needed both to project RealSense depth into the event-camera frame and to place
all event-camera observations in a common world coordinate system.

The recorded dataset contains **48 object-specific sequences** of static,
colored building-block structures on a mostly textureless white table. For
each sequence, the robot moves the camera rig along a smooth path through
randomly sampled, reachable viewpoints on a restricted hemisphere around the
object. Each trajectory is constructed from 30 target poses and is recorded
continuously, producing synchronized event streams, RGB-D measurements, and
camera poses. The split contains **42 training sequences** and **6 held-out
evaluation sequences**; there is no separate test set.

The RealSense depth measurements serve as supervision rather than model input:
after calibration they are projected into the event-camera frame to form
ground-truth depth maps. The event camera records at 1280 x 720 pixels, while
RealSense depth is captured at 640 x 480 pixels and 30 Hz. An optical filter in
front of the event-camera lens suppresses events caused by the RealSense
infrared projector.

## End-to-end reconstruction pipeline

The project covers all stages from physical recording to a reconstructed
surface. This branch starts from already recorded data (stage 2); the recording
and calibration code is on the `main` branch.

1. **Synchronized recording.** The robot follows a multi-view trajectory around
   a static object. Event data, RealSense RGB-D frames, and end-effector poses
   are recorded and associated through hardware triggering and ZeroMQ-based
   synchronization.
2. **Calibration and preprocessing.** Hand-eye and multi-camera calibration
   determine the event-camera poses and the transformation between both
   cameras. RealSense depth is projected into the event-camera frame, raw
   events are accumulated into five-bin voxel grids, and a view-dependent
   table-plane depth prior is generated.
3. **MVS-inspired depth prediction.** One event observation is selected as the
   reference view and up to eight displaced observations of the same scene are
   used as source views. A shared feature pyramid extracts multi-scale features
   from each event voxel grid and table prior. As in RGB MVS methods, calibrated
   camera geometry warps source-view features across depth hypotheses. Cascaded
   cost volumes progressively narrow the depth interval, after which a
   refinement network predicts full-resolution reference-view depth and a
   per-pixel confidence map.
4. **TSDF reconstruction.** Predicted depth maps are transformed into the
   common world frame and integrated into a truncated signed distance field.
   Confidence-weighted fusion reduces the influence of unreliable predictions,
   and the final surface is extracted as a triangle mesh.

```text
Event camera + RGB-D supervision + robot poses
                      |
                      v
       calibration and synchronized recording
                      |
                      v
 event voxel grids + table priors + ground-truth depth
                      |
                      v
 geometry-aware multi-view depth and confidence prediction
                      |
                      v
        confidence-weighted TSDF fusion -> 3-D mesh
```

## Results on the recorded dataset

The following results are measured on the six held-out recording sequences,
whose objects were not used for training. Evaluation is restricted to the
workspace cube surrounding the recording area. In this region, the
geometry-based multiview model achieves a mean absolute depth error (MAE) of
**0.180 cm** and a root mean squared error (RMSE) of **0.630 cm**.

For reconstruction, the predicted depth maps are fused with confidence-weighted
TSDF integration and compared with reference meshes reconstructed from the
ground-truth depth maps. The values below are averages over the same six
held-out objects. Accuracy measures predicted-to-reference surface distance,
completeness measures reference-to-predicted distance, and their symmetric mean
is the Chamfer distance; lower is better. Normal consistency is better when
closer to one.

Inside the workspace cube, the reconstructed meshes achieve a mean accuracy of
**0.108 cm**, completeness of **0.108 cm**, Chamfer distance of **0.108 cm**,
and normal consistency of **0.966**. These results show that the predicted
event-based depth maps remain sufficiently consistent across viewpoints to
produce coherent surfaces after fusion.

## Repository layout

| Path | Purpose |
| --- | --- |
| `3d_reconstruction/config.py`, `helpers.py` | Shared constants, camera geometry, dataset discovery, workspace masks, and view selection. |
| `3d_reconstruction/camera_data/` | Camera intrinsics, extrinsics, depth scale, and recorded end-effector poses for the original hardware setup. |
| `3d_reconstruction/data_precomputation/` | Scripts that project depth into event-camera geometry, create table priors, and build event voxel grids. |
| `3d_reconstruction/training/` | MVS depth network and training loop, evaluation, losses, and TensorBoard helpers. |
| `3d_reconstruction/reconstruction.py` | Depth prediction and TSDF reconstruction with surface and rendered-depth metrics. |
| `docker_installation/training_and_reconstruction/` | Dockerfile for the preprocessing, training, and reconstruction environment. |

## Requirements

Everything in this branch runs inside the `training_and_reconstruction` Docker
image: raw-data preprocessing, MVS-inspired training, quantitative evaluation,
and TSDF reconstruction. A
Linux machine is recommended, and model training and reconstruction require an
NVIDIA GPU with a working NVIDIA Container Toolkit installation.

Build the image from the repository root. Supplying the local user and group
IDs keeps bind-mounted outputs writable by the host user:

```bash
docker build \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  -t robot-record-reconstruction \
  docker_installation/training_and_reconstruction
```

Start an interactive container with the repository mounted at `/workspace`:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v "$(pwd):/workspace" \
  -w /workspace/3d_reconstruction \
  robot-record-reconstruction bash
```

The commands below are executed from this container shell. Adjust `--shm-size`,
batch sizes, and worker counts to the available RAM, shared memory, and GPU.

## Input data format

The directory and file names in this section are part of the data interface;
they are not placeholders except for names written inside angle brackets.
Each recording of one object is one **sequence directory**. The recorder
(`data_recording/rec_data.py` on the `main` branch) creates that directory as:

```text
3d_reconstruction/data/real/<object-name>/
├── raw_event_data/
│   └── events_cam0.raw
├── hdf5/
│   ├── realsense.h5
│   ├── events_cam0.h5
│   ├── poses.h5
│   ├── raw_poses.h5              # recorder diagnostics; not precompute input
│   └── metadata.h5               # recorder metadata; not precompute input
└── videos/                       # previews; not precompute input
    ├── events_cam0.mp4
    ├── realsense_depth.mp4
    └── realsense_rgb.mp4
```

`<object-name>` is the name entered when recording; spaces and other special
characters are replaced with underscores. The three names
`raw_event_data`, `hdf5`, and `events_cam0.raw`, and the HDF5 filenames shown
above, must remain exactly as written. Do not place the HDF5 files directly in
the object directory.

For multiview training, organize the complete sequence directories into the
following exact split layout. Moving a sequence means moving its entire
`<object-name>/` directory, without changing anything inside it:

```text
3d_reconstruction/data/<dataset-name>/
├── train/
│   ├── <training-object-1>/
│   │   ├── raw_event_data/events_cam0.raw
│   │   └── hdf5/
│   │       ├── realsense.h5
│   │       ├── events_cam0.h5
│   │       └── poses.h5
│   └── <training-object-2>/
│       └── ...
└── eval/
    ├── <evaluation-object-1>/
    │   ├── raw_event_data/events_cam0.raw
    │   └── hdf5/
    │       ├── realsense.h5
    │       ├── events_cam0.h5
    │       └── poses.h5
    └── <evaluation-object-2>/
        └── ...
```

For example, `DATA_DIR="../data/my_dataset"` in `training/train_mvs.sh`
means that the script reads sequences from `data/my_dataset/train/` and
validates on sequences from `data/my_dataset/eval/`. The split directory names
must literally be `train` and `eval`. Evaluation and reconstruction instead
receive the evaluation split itself, for example
`DATA_DIR="../data/my_dataset/eval"` in `training/eval.sh` and
`DATA_DIR="data/my_dataset/eval"` in `reconstruction.sh`.

`precompute_all.sh` accepts one or more complete sequence directories through
`--data_dir`, or searches recursively beneath `--data_root`. A directory is
recognized by voxel preprocessing as a raw sequence only when it contains both
`hdf5/realsense.h5` and at least one
`raw_event_data/events_cam*.raw` file. Therefore it is safe to run preprocessing
on the dataset root above: it will find sequences inside both splits.

The recorder-compatible file contents are described below. `N` always denotes
the number of synchronized RealSense frames in one sequence. Unless explicitly
marked optional, names, group paths, shapes, and index correspondence are
required.

### `raw_event_data/events_cam0.raw`

A Metavision-compatible RAW event stream. Events must contain native event
coordinates, timestamps in microseconds, and polarity. The recorder writes one
rising-edge external-trigger event for every synchronized RealSense frame. The
voxel preprocessor reads both events and triggers directly from this file. It
must contain at least `N` rising-edge triggers, and both timestamp types must
use the same event-camera clock.

Additional event cameras may be supplied as `events_cam1.raw`,
`events_cam2.raw`, and so on. The current training pipeline consumes
`voxels_cam0.h5`.

### `hdf5/realsense.h5`

| Dataset | Shape and type | Use |
| --- | --- | --- |
| `depth` | `(N, H_d, W_d)`, `uint16` | **Required.** Raw RealSense depth units. The conversion to metres is stored separately in `camera_data/depth_scale.npz`. |
| `rgb` | `(N, H_r, W_r, 3)`, `uint8` | Optional BGR color frames. When present, the default projection also creates `rgb_in_event_frame.h5`; omit them when only depth/event processing is needed. |
| `t_global_ms` | `(N,)`, numeric | **Required by voxel preprocessing.** RealSense global timestamps in milliseconds. Older imported recordings may instead provide `t_sys_ns`; only the array length is used. |
| `t_rgb_ms` | `(N,)`, numeric | Optional RGB timestamps retained for synchronization diagnostics. |
| `frame_number` | `(N,)`, integer | Optional RealSense frame identifiers. |

All frame-indexed datasets must be ordered consistently. The supplied recorder
uses `480 x 640` depth and RGB frames, but preprocessing reads the calibrated
resolution rather than requiring these literal dimensions.

### `hdf5/events_cam0.h5`

This is an optional frame-aligned preview generated by `rec_data.py`. It is
included in the full recorder dataset but excluded from the core dataset and
is not needed by `precompute_all.sh`. When present, it contains an `events`
group with:

| Dataset or attribute | Shape/value and use |
| --- | --- |
| `events/frames` | `(N, H_e, W_e)`, `uint8` event preview frames. |
| `events/t_ev_start_us` | `(N,)`, integer start time of every aligned event window. |
| `events/t_ev_end_us` | `(N,)`, integer end time of every aligned event window. |
| `events/hw_trigger_times_us` | `(N,)` copy of the RAW hardware triggers. |
| `events/alignment_offset_us` | Optional `(N,)` signed diagnostic offsets between selected event-window centers and triggers. |
| `events.attrs["height"]`, `events.attrs["width"]` | Native event-camera dimensions as integer attributes. |
| `events.attrs["fps"]`, `events.attrs["delta_t_us"]` | Optional preview-frame metadata written by the recorder. |
| `events.attrs["alignment_mode"]` | Optional diagnostic value `"hw_trigger"`. |

Preprocessing does not read this preview file. For an external dataset, place
the synchronized rising-edge triggers in the RAW stream itself.

### `hdf5/poses.h5`

| Dataset | Required shape and type | Meaning |
| --- | --- | --- |
| `ee_T` | `(N, 4, 4)`, floating point | One homogeneous `T_base_from_ee` end-effector pose per synchronized frame. |
| `nearest_offset_ms` | `(N,)`, floating point | Optional diagnostic time difference to the raw robot pose selected for that frame. |

The frame at index `i` in `poses.h5` must describe the camera pose for frame
`i` in `realsense.h5`; rising-edge trigger `i` in the RAW event stream defines
the corresponding event window. In a normal recording, `depth`, `rgb`,
`t_global_ms`, `ee_T`, and `nearest_offset_ms` therefore have the same leading
length `N`. When the optional `events_cam0.h5` preview is present, its
frame-indexed datasets also have length `N`. Preprocessing can truncate
depth/table generation to an available minimum in some mismatch cases, but
such a sequence is not the intended input format and should be repaired before
training.

The supplied recorder additionally creates `raw_poses.h5`, `metadata.h5`, and
preview videos. These are useful for provenance and diagnostics but are not
inputs to `precompute_all.sh`.

### Calibration files

Place the calibration in `3d_reconstruction/camera_data/`. To use another
location for the complete pipeline, update `CALIB_DIR` in
`3d_reconstruction/config.py`; the projection script's `--calib_dir` flag only
changes the projection stage. The complete pipeline expects:

| File | Required arrays |
| --- | --- |
| `event_intrinsics.npz` | `camera_matrix` `(3,3)`, `dist_coeffs`, and `image_size` `[W,H]` |
| `rs_depth_intrinsics.npz` | `camera_matrix` `(3,3)` and `image_size` `[W,H]` |
| `rs_rgb_intrinsics.npz` | `camera_matrix` `(3,3)` and `image_size` `[W,H]` |
| `depth_scale.npz` | Scalar `scale`, converting stored `uint16` depth to metres |
| `T_event_from_depth.npz` | Homogeneous transform `T` `(4,4)` |
| `T_color_from_depth.npz` | Homogeneous transform `T` `(4,4)` |
| `T_event_from_rgb.npz` | Homogeneous transform `T` `(4,4)` |
| `T_rgb_from_ee.npz` | Homogeneous transform `T` `(4,4)` |

The last two transforms are composed to obtain the event-camera pose relative
to the robot end effector for table-prior generation, training, evaluation,
and reconstruction.

## Preprocessing

From `/workspace/3d_reconstruction` inside the Docker container, process an
entire dataset with:

```bash
./data_precomputation/precompute_all.sh --data_root data/my_dataset
```

Or process one or more explicit sequences:

```bash
./data_precomputation/precompute_all.sh \
  --data_dir data/my_dataset/train/object_01 data/my_dataset/eval/object_07
```

The script runs depth/RGB projection, table-plane prior generation, and event
voxel generation in the required order. It does not create a new sequence
directory; it adds the derived files to each existing sequence. After a
successful run, the complete sequence layout is:

```text
<sequence>/
├── raw_event_data/
│   └── events_cam0.raw             # original input
├── events/
│   └── voxels_cam0.h5              # generated
│       ├── /voxels                 # (N, 5, 240, 320), float32 by default
│       └── /hw_trigger_times_us     # (N,), integer microseconds
└── hdf5/
    ├── realsense.h5                 # original input
    ├── events_cam0.h5               # optional recorder preview; not in core_dataset
    ├── poses.h5                     # original input
    ├── depth_in_event_frame.h5      # generated: /depth, (N,240,320), float32 metres
    ├── rgb_in_event_frame.h5        # generated if RGB exists: /rgb, (N,240,320,3), uint8
    └── table_plane.h5               # generated: /table_plane, (N,240,320), float32 [0,1]
```

All spatial products use the canonical native center crop to `720 x 960`
followed by resizing to `240 x 320`. `precompute_all.sh` overwrites derived
files. Its `--project`, `--table`, and `--voxel` section markers can be used to
forward stage-specific options; run each underlying script with `--help` for
the available settings.

A sequence is ready for training only when these four exact paths exist:

```text
<sequence>/events/voxels_cam0.h5
<sequence>/hdf5/depth_in_event_frame.h5
<sequence>/hdf5/poses.h5
<sequence>/hdf5/table_plane.h5
```

The model loader uses the dataset names `/voxels`, `/depth`, `/ee_T`, and
`/table_plane` inside those files. `rgb_in_event_frame.h5` is useful for
visualization but is not a model input. Frame index `i` must refer to the same
instant and camera pose in all four model inputs.

### Extracting distributable datasets

To extract both dataset variants from the default `data/new_2` source, run:

```bash
python3 data_precomputation/extract_dataset.py
```

This creates `data/dataset`, containing only files written by `rec_data.py`,
and `data/core_dataset`, containing only the three per-recording source files
needed to run the complete precomputation pipeline:

```text
<sequence>/raw_event_data/events_cam0.raw
<sequence>/hdf5/realsense.h5
<sequence>/hdf5/poses.h5
```

The core dataset deliberately excludes derived files such as
`hdf5/events_cam0.h5`, `events/voxels_cam0.h5`,
`hdf5/depth_in_event_frame.h5`, and `hdf5/table_plane.h5`. Both outputs
preserve the source's split and sequence hierarchy. The extractor validates
every sequence before copying and refuses to replace an existing output unless
`--overwrite` is supplied. Use `--dry-run` to validate and report the required
storage without writing files.

If `data/dataset` is already correct and only the core export must be rebuilt,
use `--core-only --overwrite`. This replaces `data/core_dataset` without
touching `data/dataset`.

On Slurm, submit the equivalent launcher:

```bash
sbatch data_precomputation/extract_dataset.sbatch
```

The source and both output paths can be edited near the top of the launcher or
overridden with command-line arguments.

## Local training, evaluation, and reconstruction

The configuration blocks of the local `.sh` launchers are deliberately near the
top of each file. Edit the dataset paths, run names, checkpoints, batch sizes,
workers, and model options there before running them.

Train the MVS-inspired model:

```bash
./training/train_mvs.sh
```

Evaluate a trained checkpoint:

```bash
./training/eval.sh
```

Create TSDF reconstructions and compare uniform with confidence-weighted
fusion:

```bash
./reconstruction.sh
```

Each launcher also appends arguments supplied on the command line, making short
temporary overrides possible without editing the file. For example:

```bash
./training/train_mvs.sh --epochs 5 --name local_smoke_test
```

The scripts assume they are already running inside the main Docker image. They
resolve the repository location from their own path, so they can be invoked
from any working directory. Training outputs, evaluation reports, and
reconstruction meshes are written into the mounted repository and therefore
remain available after the container exits.
