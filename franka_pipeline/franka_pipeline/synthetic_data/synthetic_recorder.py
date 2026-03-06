"""
Synthetic data recorder for simulation environments.

Records RGB, depth, events, poses, and camera parameters from simulated environments.
Output format is compatible with the multi_recording.py real data format.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import h5py
import cv2

from .rgb_to_events import RGBToEventsConverter, EventArray, events_to_frame
from franka_pipeline.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticRecorderConfig:
    """Configuration for synthetic data recording."""
    
    # Output directories
    output_dir: Path = field(default_factory=lambda: Path("synthetic_data"))
    
    # Camera settings
    camera_id: str = "robot0_eye_in_hand"  # Which camera to use from simulation
    camera_width: int = 640
    camera_height: int = 480
    
    # Event camera simulation settings
    event_threshold_pos: float = 0.25
    event_threshold_neg: float = 0.25
    event_refractory_us: int = 1000
    event_leak_rate: float = 0.0
    event_shot_noise_rate: float = 0.0
    
    # Recording settings
    save_rgb: bool = True
    save_depth: bool = True
    save_events: bool = True
    save_poses: bool = True
    save_video: bool = True
    video_fps: int = 30
    
    # Image flip settings (useful for camera mounting orientation)
    flip_vertical: bool = False  # Flip all images and event y-coordinates vertically


class SyntheticDataRecorder:
    """
    Records synthetic sensor data from simulation environments.
    
    Produces data in a format compatible with real recordings from multi_recording.py:
    - HDF5 files with depth frames and timestamps
    - Synthetic events in structured numpy format
    - Robot poses synchronized with sensor data
    - Optional RGB video for visualization
    """
    
    def __init__(self, config: SyntheticRecorderConfig):
        self.config = config
        self.is_recording = False
        self.recording_id: Optional[str] = None
        
        # Data buffers
        self._rgb_frames: list[np.ndarray] = []
        self._depth_frames: list[np.ndarray] = []
        self._timestamps_ns: list[int] = []
        self._poses: list[Dict[str, Any]] = []
        self._events: list[EventArray] = []
        
        # Event converter
        self._event_converter: Optional[RGBToEventsConverter] = None
        
        # Video writers
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._depth_video_writer: Optional[cv2.VideoWriter] = None
        
        # Frame counter
        self._frame_idx = 0
    
    def start_recording(self, recording_id: Optional[str] = None) -> None:
        """Start a new recording session."""
        if self.is_recording:
            logger.warning("Already recording, stopping previous recording first")
            self.stop_recording()
        
        self.recording_id = recording_id or f"recording_{int(time.time())}"
        
        # Create output directories
        self._output_path = self.config.output_dir / self.recording_id
        self._output_path.mkdir(parents=True, exist_ok=True)
        
        (self._output_path / "hdf5").mkdir(exist_ok=True)
        (self._output_path / "events").mkdir(exist_ok=True)
        (self._output_path / "videos").mkdir(exist_ok=True)
        (self._output_path / "poses").mkdir(exist_ok=True)
        
        # Reset buffers
        self._rgb_frames = []
        self._depth_frames = []
        self._timestamps_ns = []
        self._poses = []
        self._events = []
        self._frame_idx = 0
        
        # Initialize event converter
        self._event_converter = RGBToEventsConverter(
            threshold_pos=self.config.event_threshold_pos,
            threshold_neg=self.config.event_threshold_neg,
            refractory_period_us=self.config.event_refractory_us,
            leak_rate=self.config.event_leak_rate,
            shot_noise_rate=self.config.event_shot_noise_rate,
        )
        
        # Initialize video writers
        if self.config.save_video:
            video_path = self._output_path / "videos" / "rgb.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.config.video_fps,
                (self.config.camera_width, self.config.camera_height),
                isColor=True,
            )
            
            # Depth video writer (grayscale visualization)
            depth_video_path = self._output_path / "videos" / "depth.mp4"
            self._depth_video_writer = cv2.VideoWriter(
                str(depth_video_path),
                fourcc,
                self.config.video_fps,
                (self.config.camera_width, self.config.camera_height),
                isColor=False,
            )
        
        self.is_recording = True
        logger.info(f"Started synthetic recording: {self.recording_id}")
    
    def record_frame(
        self,
        sim_env,
        robot_state: Dict[str, Any],
        timestamp_ns: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Record a single frame from the simulation environment.
        
        Args:
            sim_env: Simulation environment (must have get_image method)
            robot_state: Current robot state dictionary
            timestamp_ns: Optional timestamp, defaults to current time
            
        Returns:
            Dictionary with recording stats
        """
        if not self.is_recording:
            return {"error": "Not recording"}
        
        timestamp_ns = timestamp_ns or time.time_ns()
        timestamp_us = timestamp_ns // 1000
        
        stats = {"frame_idx": self._frame_idx, "num_events": 0}
        
        # Get RGB image
        rgb_image = None
        if self.config.save_rgb or self.config.save_events:
            rgb_image = sim_env.get_image(
                self.config.camera_id, rgb=True, depth=False
            )
            if rgb_image is not None:
                # Ensure correct shape and type
                if rgb_image.dtype != np.uint8:
                    rgb_image = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
                
                # Apply vertical flip if configured
                if self.config.flip_vertical:
                    rgb_image = np.flipud(rgb_image).copy()
                
                if self.config.save_rgb:
                    self._rgb_frames.append(rgb_image.copy())
                
                # Write to video
                if self._video_writer is not None:
                    # OpenCV expects BGR
                    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    self._video_writer.write(bgr)
        
        # Get depth image
        if self.config.save_depth:
            depth_image = sim_env.get_image(
                self.config.camera_id, rgb=False, depth=True
            )
            if depth_image is not None:
                # Convert depth to uint16 millimeters for compatibility with RealSense format
                if depth_image.ndim == 3:
                    depth_image = depth_image[:, :, 0]
                depth_mm = (depth_image * 1000).astype(np.uint16)
                
                # Apply vertical flip if configured
                if self.config.flip_vertical:
                    depth_mm = np.flipud(depth_mm).copy()
                
                self._depth_frames.append(depth_mm)
                
                # Write depth visualization to video
                if self._depth_video_writer is not None:
                    # Scale depth for visualization (0-5m range -> 0-255)
                    depth_vis = np.clip(depth_mm / 5000.0 * 255, 0, 255).astype(np.uint8)
                    self._depth_video_writer.write(depth_vis)
        
        # Generate synthetic events from RGB
        # Note: rgb_image is already flipped if flip_vertical is enabled,
        # so events generated from it are automatically in the correct coordinate space
        if self.config.save_events and rgb_image is not None:
            events = self._event_converter.convert_frame(rgb_image, timestamp_us)
            if len(events) > 0:
                self._events.append(events)
            stats["num_events"] = len(events)
        
        # Save pose data
        if self.config.save_poses:
            pose_data = {
                "t_ns": timestamp_ns,
                "frame_idx": self._frame_idx,
            }
            
            # Extract relevant pose information
            if "osc_pose" in robot_state:
                pose_data["ee_pose"] = robot_state["osc_pose"].tolist()
            if "joint_position" in robot_state:
                pose_data["joint_position"] = robot_state["joint_position"].tolist()
            if "gripper_q" in robot_state:
                pose_data["gripper_q"] = float(np.asarray(robot_state["gripper_q"]).flatten()[0])
            
            # Get camera parameters
            intrinsics = sim_env.get_camera_intrinsics(self.config.camera_id)
            if intrinsics:
                pose_data["camera_intrinsics"] = intrinsics
            
            extrinsics = sim_env.get_camera_extrinsics(self.config.camera_id)
            if extrinsics is not None:
                pose_data["camera_extrinsics"] = extrinsics.tolist()
            
            self._poses.append(pose_data)
        
        self._timestamps_ns.append(timestamp_ns)
        self._frame_idx += 1
        
        return stats
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording and save all data to disk.
        
        Returns:
            Dictionary with recording summary
        """
        if not self.is_recording:
            return {"error": "Not recording"}
        
        logger.info(f"Stopping recording: {self.recording_id}")
        
        summary = {
            "recording_id": self.recording_id,
            "num_frames": self._frame_idx,
            "output_path": str(self._output_path),
        }
        
        # Close video writers
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            summary["video_saved"] = True
        
        if self._depth_video_writer is not None:
            self._depth_video_writer.release()
            self._depth_video_writer = None
            summary["depth_video_saved"] = True
        
        # Save depth data to HDF5 (compatible with multi_recording.py format)
        if self.config.save_depth and self._depth_frames:
            h5_path = self._output_path / "hdf5" / "depth.h5"
            with h5py.File(h5_path, "w") as f:
                grp = f.create_group("realsense")  # Keep same group name for compatibility
                
                depth_array = np.stack(self._depth_frames, axis=0)
                grp.create_dataset("depth", data=depth_array, compression="gzip")
                
                timestamps = np.array(self._timestamps_ns, dtype=np.int64)
                grp.create_dataset("t_sys_ns", data=timestamps)
                
                # Add metadata
                grp.attrs["width"] = self.config.camera_width
                grp.attrs["height"] = self.config.camera_height
                grp.attrs["synthetic"] = True
            
            summary["depth_h5_saved"] = str(h5_path)
            logger.info(f"Saved depth data: {len(self._depth_frames)} frames")
        
        # Save events to numpy file
        if self.config.save_events and self._events:
            # Concatenate all events
            all_x = np.concatenate([e.x for e in self._events])
            all_y = np.concatenate([e.y for e in self._events])
            all_t = np.concatenate([e.t for e in self._events])
            all_p = np.concatenate([e.p for e in self._events])
            
            # Save as structured array (compatible with Metavision format)
            dtype = np.dtype([
                ('x', '<u2'),
                ('y', '<u2'),
                ('p', '<i2'),
                ('t', '<i8')
            ])
            events_array = np.zeros(len(all_x), dtype=dtype)
            events_array['x'] = all_x
            events_array['y'] = all_y
            events_array['p'] = all_p.astype(np.int16)
            events_array['t'] = all_t
            
            events_path = self._output_path / "events" / "events.npy"
            np.save(events_path, events_array)
            
            summary["events_saved"] = str(events_path)
            summary["total_events"] = len(events_array)
            logger.info(f"Saved events: {len(events_array)} total events")
            
            # Also save event visualization video
            if self.config.save_video and len(self._rgb_frames) > 0:
                self._save_event_video()
        
        # Save poses to numpy file
        if self.config.save_poses and self._poses:
            poses_path = self._output_path / "poses" / "poses.npy"
            np.save(poses_path, self._poses, allow_pickle=True)
            summary["poses_saved"] = str(poses_path)
            logger.info(f"Saved poses: {len(self._poses)} entries")
        
        # Save RGB frames to HDF5 (optional, for event conversion debugging)
        if self.config.save_rgb and self._rgb_frames:
            rgb_h5_path = self._output_path / "hdf5" / "rgb.h5"
            with h5py.File(rgb_h5_path, "w") as f:
                rgb_array = np.stack(self._rgb_frames, axis=0)
                f.create_dataset("rgb", data=rgb_array, compression="gzip")
                f.create_dataset("t_sys_ns", data=np.array(self._timestamps_ns, dtype=np.int64))
            summary["rgb_h5_saved"] = str(rgb_h5_path)
        
        # Reset state
        self.is_recording = False
        self.recording_id = None
        self._rgb_frames = []
        self._depth_frames = []
        self._timestamps_ns = []
        self._poses = []
        self._events = []
        self._event_converter = None
        
        logger.info(f"Recording complete. Summary: {summary}")
        return summary
    
    def _save_event_video(self) -> None:
        """Save a visualization video of the synthetic events.
        
        Creates one frame per recorded simulation frame (matching RGB/depth video length),
        with events binned by their corresponding frame timestamps.
        """
        if not self._events or not self._timestamps_ns:
            return
        
        video_path = self._output_path / "videos" / "events.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.config.video_fps,
            (self.config.camera_width, self.config.camera_height),
            isColor=True,
        )
        
        # Concatenate all events
        all_events = EventArray(
            x=np.concatenate([e.x for e in self._events]),
            y=np.concatenate([e.y for e in self._events]),
            t=np.concatenate([e.t for e in self._events]),
            p=np.concatenate([e.p for e in self._events]),
        )
        
        # Convert timestamps to microseconds (events use microseconds)
        timestamps_us = [t_ns // 1000 for t_ns in self._timestamps_ns]
        
        # Create one frame per recorded simulation frame (matches RGB/depth frame count)
        for i in range(len(timestamps_us)):
            # Get time boundaries for this frame
            t_start = timestamps_us[i - 1] if i > 0 else (timestamps_us[0] - 1)
            t_end = timestamps_us[i]
            
            # Get events in this time window
            mask = (all_events.t > t_start) & (all_events.t <= t_end)
            frame_events = EventArray(
                x=all_events.x[mask],
                y=all_events.y[mask],
                t=all_events.t[mask],
                p=all_events.p[mask],
            )
            
            frame = events_to_frame(
                frame_events,
                self.config.camera_height,
                self.config.camera_width,
            )
            
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        
        writer.release()
        logger.info(f"Saved event visualization video: {video_path} ({len(timestamps_us)} frames)")
