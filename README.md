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

> **Safety notice:** this code can command a physical robot. Run new code in
> simulation first, keep the emergency stop accessible, verify workspace and
> collision limits, and never operate the real robot unattended. The included
> calibration files and workspace constants describe one particular setup and
> must be revalidated for any other robot, camera mount, table, or scene.

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
whose objects were not used for training. Depth errors are reported in three
regions: the whole image, a workspace cube around the recording area, and a
raised cube that removes most of the dominant table surface and therefore
focuses the evaluation on the object itself. Lower MAE and RMSE are better.

<p align="center">
  <img src="docs/images/final_results_table%20-%20Kopie.png" alt="Depth-prediction results on the held-out recorded dataset" width="560">
</p>

<p align="center"><em>Mean depth-prediction error of the geometry-based multiview model over the held-out sequences.</em></p>

The model reaches a whole-frame MAE of **0.182 cm** and a workspace-cube MAE
of **0.180 cm**. The raised-cube MAE is higher at **0.522 cm**, because this
region emphasizes the more difficult object surfaces and depth discontinuities
instead of the large, comparatively simple table.

For reconstruction, the predicted depth maps are fused with confidence-weighted
TSDF integration and compared with reference meshes reconstructed from the
ground-truth depth maps. The values below are averages over the same six
held-out objects. Accuracy measures predicted-to-reference surface distance,
completeness measures reference-to-predicted distance, and their symmetric mean
is the Chamfer distance; lower is better. Normal consistency is better when
closer to one.

<p align="center">
  <img src="docs/images/mesh_results_table%20-%20Kopie.png" alt="TSDF mesh-reconstruction results on the held-out recorded dataset" width="820">
</p>

<p align="center"><em>Mean surface metrics for confidence-weighted TSDF reconstructions of the held-out sequences.</em></p>

Inside the workspace cube, the reconstructed meshes achieve a mean Chamfer
distance of **0.108 cm** and normal consistency of **0.966**. In the more
object-focused raised cube, the Chamfer distance is **0.180 cm** and normal
consistency is **0.905**. These results show that the predicted event-based
depth maps remain sufficiently consistent across viewpoints to produce coherent
surfaces after fusion.

## Repository layout

| Path | Purpose |
| --- | --- |
| `franka_pipeline/` | Franka robot control, simulation, teleoperation agents, pose streaming, and synthetic-data collection. |
| `3d_reconstruction/` | Recording client, calibration, preprocessing, training, evaluation, reconstruction, and visualization tools. |
| `3d_reconstruction/camera_data/` | Example camera intrinsics, extrinsics, depth scale, and recorded end-effector poses for the original hardware setup. |
| `3d_reconstruction/data_precomputation/` | Scripts that project depth into event-camera geometry, create table priors, and build event voxel grids. |
| `3d_reconstruction/training/` | Multiview depth network, training loop, evaluation, and TensorBoard helpers. |
| `3d_reconstruction/viz_and_tests/` | Analysis and visualization utilities; these are primarily research diagnostics. |
| `docker_installation/` | Dockerfiles for the robot backend, robot frontend, and recording/training environment. |
| `instructions/` | Hardware notes and vendor documentation retained from the development setup. |

## Software architecture

The physical-data workflow uses two cooperating processes:

1. `franka_pipeline/main.py` controls the robot or simulator and publishes
   timestamped end-effector poses plus episode events through ZeroMQ.
2. `3d_reconstruction/rec_data.py` records the cameras, subscribes to those
   poses, and uses a request/reply handshake to synchronize recording.

Shared recording, preprocessing, and reconstruction constants live in
[`3d_reconstruction/config.py`](3d_reconstruction/config.py), including image
sizes, voxel-bin count, workspace dimensions, depth limits, and local ZeroMQ
addresses.

## Requirements

The full real-robot workflow was developed for Linux, NVIDIA GPU acceleration,
Docker, a Franka Emika Panda, Deoxys, an Intel RealSense camera, and an
event-camera stack based on Metavision/IDS uEye EVS. It is hardware-specific;
this repository does not make the system plug-and-play on arbitrary hardware.

For reconstruction and training, the recording/training Dockerfile installs the
main Python packages: PyTorch, Open3D, OpenCV, NumPy, SciPy, h5py, pyzmq,
msgpack, TensorBoard, Matplotlib, and related tools. Build it from its directory:

```bash
cd docker_installation/training_and_reconstruction
docker build -t robot-record-reconstruction .
```

For the robot pipeline, install the Python dependencies listed in
[`franka_pipeline/requirements.txt`](franka_pipeline/requirements.txt), plus
the hardware-specific packages that are intentionally left optional there
(`pyrealsense2`, OpenCV, input-device packages, Deoxys, and robosuite). The
robot backend and frontend Dockerfiles document the original container setup.

The Dockerfiles download third-party dependencies at build time and may need
updates as upstream package repositories change. They also use privileged,
host-networked container settings for hardware access; review those settings
before running them on a shared system.

## Quick start: simulation and synthetic data

Simulation is the recommended first step. From `franka_pipeline/`, inspect the
available options and objects:

```bash
python main.py --help
python main.py --list-objects
```

Run a headless synthetic-data collection job, optionally restricting the set of
objects:

```bash
python main.py \
  --simulated-robot \
  --synthetic-data \
  --headless \
  --object-filter cube,sphere \
  --synthetic-output-dir data/synthetic
```

The runner supports hemisphere and random-hemisphere camera trajectories. Use
`--num-poses`, `--sphere-radius`, `--target-x`, `--target-y`, `--target-z`,
and `--random-seed` to control them. Exact defaults are in
[`franka_pipeline/config_defaults.py`](franka_pipeline/config_defaults.py).

## Physical recording

Only use this section after hardware, network addresses, calibration, and
emergency-stop procedures have been checked.

Start the robot-side pipeline with synchronized recording enabled:

```bash
cd franka_pipeline
python main.py --real-robot --sync-recording
```

Then start the recording client in a separate environment:

```bash
cd 3d_reconstruction
python rec_data.py \
  --zmq-sync-addr tcp://ROBOT_HOST:6001 \
  --zmq-pose-addr tcp://ROBOT_HOST:6000
```

The default addresses are localhost ports `6000` and `6001`. Change them for
separate machines and expose only trusted interfaces. The recording client
expects the required camera drivers and the physical cameras to be available.

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

This project depends on third-party systems including Deoxys, robosuite,
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
