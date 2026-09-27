# Robot Recording and Event-Based 3-D Reconstruction

This repository contains the complete research pipeline developed for my
master's thesis: synchronized data recording with robot-mounted cameras,
geometric preprocessing, event-based multi-view depth prediction, and dense
3-D reconstruction through TSDF fusion.

**[Read the master's thesis (PDF)](docs/Masters_Thesis_github.pdf)**

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

<p align="center">
  <img src="docs/images/cameras_2%20-%20Kopie.jpg" alt="Close-up of the synchronized event and RGB-D cameras mounted on the robot end effector" width="760">
</p>

<p align="center"><em>The event camera and RealSense depth camera on the custom end-effector mount.</em></p>

<p align="center">
  <img src="docs/images/robot_arm_2%20-%20Kopie.jpg" alt="Franka robot recording a building-block object on the tabletop" width="560">
</p>

<p align="center"><em>The complete recording setup with a static building-block object in the workspace.</em></p>

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
surface:

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
| `franka_pipeline/` | Franka robot control, trajectory execution, pose streaming, and synchronized data collection. |
| `3d_reconstruction/` | Recording client, calibration, preprocessing, training, evaluation, reconstruction, and visualization tools. |
| `3d_reconstruction/camera_data/` | Example camera intrinsics, extrinsics, depth scale, and recorded end-effector poses for the original hardware setup. |
| `3d_reconstruction/data_precomputation/` | Scripts that project depth into event-camera geometry, create table priors, and build event voxel grids. |
| `3d_reconstruction/training/` | Multiview depth network, training loop, evaluation, and TensorBoard helpers. |
| `3d_reconstruction/viz_and_tests/` | Analysis and visualization utilities; these are primarily research diagnostics. |
| `docker_installation/` | Dockerfiles for the robot backend, robot frontend, and recording/training environment. |
| `instructions/` | Hardware notes and vendor documentation retained from the development setup. |

## Requirements

The full real-robot workflow was developed for Linux, NVIDIA GPU acceleration,
Docker, a Franka Emika Panda, Deoxys, an Intel RealSense camera, and an
event-camera stack based on Metavision/IDS uEye EVS. It is hardware-specific;
this repository does not make the system plug-and-play on arbitrary hardware.

All scripts in `3d_reconstruction/` are intended to run inside the environment
defined by
[`docker_installation/training_and_reconstruction/Dockerfile`](docker_installation/training_and_reconstruction/Dockerfile).
This includes camera recording, calibration, data preprocessing, model training,
evaluation, and TSDF reconstruction. Build the image from its directory:

```bash
cd docker_installation/training_and_reconstruction
docker build -t robot-record-reconstruction .
```

Data recording additionally requires `franka_pipeline/` and its two-container
robot-control environment:

- The image defined by
  [`docker_installation/robot_arm_backend/Dockerfile`](docker_installation/robot_arm_backend/Dockerfile)
  runs the low-level Deoxys backend that communicates with the robot. This
  container must already be running in the background before robot commands are
  issued.
- The scripts in `franka_pipeline/` run inside the image defined by
  [`docker_installation/robot_arm_frontend/Dockerfile`](docker_installation/robot_arm_frontend/Dockerfile).
  The frontend connects to the running backend and handles trajectory execution,
  robot poses, and synchronization with the recording process.

Consequently, physical data collection uses three cooperating components: the
robot backend container, the robot frontend container running
`franka_pipeline/`, and the training-and-reconstruction container running the
recording script from `3d_reconstruction/`.

The Dockerfiles download third-party dependencies at build time and may need
updates as upstream package repositories change. They also use privileged,
host-networked container settings for hardware access; review those settings
before running them on a shared system.

If a recorded and preprocessed dataset is already available, skip the
**Calibration** and **Physical recording** sections and continue directly with
[Training](#training) or [Evaluation and reconstruction](#evaluation-and-reconstruction).

## Calibration

The repository includes calibration outputs for its original rig. Treat them as
examples, not portable parameters. Recalibrate after changing a camera, lens,
mount, robot base, table, resolution, or preprocessing transform.

To estimate calibration from existing captures:

```bash
cd 3d_reconstruction
python calibration.py --output-dir camera_data
```

To collect a new calibration sequence through the robot-side ZeroMQ service:

```bash
python calibration.py \
  --collect-data \
  --output-dir camera_data \
  --zmq-bind tcp://0.0.0.0:6002
```

Use `verify_calibration.py` and the tools in `viz_and_tests/` to inspect the
result before collecting a dataset or training a model.

## Physical recording

Only use this section after hardware, network addresses, calibration, and
emergency-stop procedures have been checked.

### Software architecture

Physical recording uses two synchronized application processes in addition to
the low-level robot backend:

1. `franka_pipeline/main.py`, running in the robot-frontend container, controls
   the robot and publishes timestamped end-effector poses and episode events
   through ZeroMQ.
2. `3d_reconstruction/rec_data.py`, running in the
   training-and-reconstruction container, records both cameras, subscribes to
   the robot poses, and uses a request/reply handshake to synchronize recording.

The Deoxys backend container communicates directly with the robot and must
remain active while the frontend process is running.

Shared recording, preprocessing, and reconstruction constants live in
[`3d_reconstruction/config.py`](3d_reconstruction/config.py), including image
sizes, voxel-bin count, workspace dimensions, depth limits, and local ZeroMQ
addresses.

### Starting a recording

First start the Deoxys backend using the robot-backend Docker image and leave it
running in the background. Then open the robot-frontend container and start the
robot-side pipeline with synchronized recording enabled:

```bash
cd franka_pipeline
python main.py --real-robot --sync-recording
```

In parallel, use the training-and-reconstruction container to start the camera
recording client:

```bash
cd 3d_reconstruction
python rec_data.py \
  --zmq-sync-addr tcp://ROBOT_HOST:6001 \
  --zmq-pose-addr tcp://ROBOT_HOST:6000
```

The default addresses are localhost ports `6000` and `6001`. Change them for
separate machines and expose only trusted interfaces. The recording client
expects the required camera drivers and the physical cameras to be available.

## Data layout and preprocessing

Raw data is intentionally ignored by Git. By default, scripts expect datasets
under `3d_reconstruction/data/real/`. A recording sequence must provide the
modalities and metadata expected by the preprocessing scripts; inspect their
`--help` output and the relevant loader code before adapting a new dataset.

The normal preprocessing order is:

1. Project RealSense depth into the event-camera reference frame.
2. Compute or load the table-plane prior.
3. Convert event streams to temporal voxel grids.

Run all three in order:

```bash
cd 3d_reconstruction
./data_precomputation/precompute_all.sh --data_root data/real
```

For a specific sequence or a crop-then-resize setup:

```bash
./data_precomputation/precompute_all.sh \
  --data_dir data/real/my_sequence \
  --crop_then_resize
```

`precompute_all.sh` forwards common flags to every stage and accepts
stage-specific flags after `--project`, `--table`, or `--voxel`. Run each Python
script with `--help` for the authoritative set of options.

## Training

The depth model consumes event voxel grids, table-plane priors, camera geometry,
and one or more selected views. Train from the reconstruction directory:

```bash
cd 3d_reconstruction
python training/train_unet.py \
  --data_dir data/real \
  --out_dir checkpoints/run_001 \
  --epochs 50 \
  --batch_size 64 \
  --num_views 1
```

For multiview training, increase `--num_views` and use `--view_interval` or
`--pose_view_selection`. Other useful switches include `--predict_uncertainty`,
`--recurrent`, `--crop_then_resize`, and loss-weight parameters. Outputs,
TensorBoard logs, checkpoints, and result directories are ignored by Git.

## Evaluation and reconstruction

Evaluate a checkpoint with:

```bash
cd 3d_reconstruction
python training/evaluation.py --help
```

Reconstruct a sequence and fuse predicted depth maps into a TSDF volume with:

```bash
python reconstruction.py --help
```

Use the command help for required positional inputs and checkpoint arguments;
these scripts have many experimental switches for frame selection, uncertainty
filtering, pose layout, workspace cube bounds, mesh extraction, and rendering.
The defaults are derived from `config.py` and are specific to the original
tabletop scene.

## Development notes

- Run Python commands from `franka_pipeline/` or `3d_reconstruction/` as shown;
  several scripts use relative imports and paths.
- The project is research code, with experimental scripts and setup-specific
  constants. There is no automated test suite or package installer at present.
- Large raw recordings, model checkpoints, meshes, plots, and logs should stay
  outside Git or be released through an archival/data service rather than
  committed to the repository.
- `viz_and_tests/` is a useful source of examples, but many scripts assume a
  particular directory structure or development dataset.

## Third-party software and licensing

This project depends on third-party systems including Deoxys,
RealSense/librealsense, Metavision, IDS uEye EVS, Open3D, PyTorch, and optional
vision-language/grasping services. Their licences, terms, and redistribution
rules apply independently.

No licence file is currently included for this repository. Before publishing,
add a `LICENSE` that expresses the permissions you intend to grant, and verify
that all bundled vendor packages and PDF documentation may legally be
redistributed. In particular, consider removing the `.deb` installers and
vendor PDFs and linking to their official download pages instead.

## Citation

For the scientific motivation, method, and complete evaluation, please refer to
the [master's thesis](docs/Masters_Thesis_github.pdf). If you use the code,
please also cite the repository URL and the commit or release version used.
