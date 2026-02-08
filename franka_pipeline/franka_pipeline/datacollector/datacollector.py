"""Data collection handler for saving robot trajectories in LeRobot format.

# inspired by  https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py


This module provides the DataCollector class for recording robot observations,
actions, and state data in the LeRobot dataset format compatible with
Hugging Face Hub.
"""

import shutil
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


class DataCollector:
    """Data collector for saving robot trajectories in LeRobot format.

    Collects observations, actions, and robot state during teleoperation
    or autonomous execution and saves them as LeRobot datasets compatible
    with Hugging Face Hub.

    Attributes:
        repo_name: The repository identifier for the dataset.
        output_path: Path where the dataset is stored.
        dataset: The underlying LeRobotDataset instance.
    """

    def __init__(
        self,
        repo_name: str = "your_hf_username/my_dataset",
        overwrite: bool = False,
        camera_width: int = 256,
        camera_height: int = 256,
    ) -> None:
        """Initialize the data collector.

        Args:
            repo_name: Repository identifier for the LeRobot dataset. Used
                for both local storage and Hugging Face Hub uploads.
            overwrite: If True, delete existing dataset at output_path.
                If False, raise ValueError if path exists.
            camera_width: Width of the camera images to store.
            camera_height: Height of the camera images to store.

        Raises:
            ValueError: If output path exists and overwrite is False.
        """
        self.repo_name = repo_name
        self.output_path = HF_LEROBOT_HOME / self.repo_name
        self.camera_width = camera_width
        self.camera_height = camera_height

        if self.output_path.exists():
            if overwrite:
                shutil.rmtree(self.output_path)
            else:
                raise ValueError(
                    f"Data folder {self.output_path} already exists. "
                    "Set overwrite=True or choose a new folder name."
                )

        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_name,
            robot_type="panda",
            fps=20,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (self.camera_height, self.camera_width, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (self.camera_height, self.camera_width, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["observation.state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def collect(
        self,
        obs: dict[str, Any],
        action: np.ndarray,
        robot_state: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """Add a single frame of data to the current episode.

        Args:
            obs: Observation dictionary containing camera images.
                Expected keys: "image", "wrist_image".
            action: Action array of shape (7,) with arm and gripper commands.
            robot_state: Robot state array of shape (8,) with joint
                positions and gripper state.
            metadata: Additional metadata including "instruction" for the task.
        """
        self.dataset.add_frame(
            {
                "image": obs.get("image"),
                "wrist_image": obs.get("wrist_image"),
                "observation.state": robot_state,
                "action": action,
                "task": metadata.get("instruction", ""),
            }
        )

    def save(self) -> None:
        """Save the current episode data to disk and start a new episode."""
        self.dataset.save_episode()

    def finalize(self) -> None:
        """Finalize the dataset after all episodes are collected."""
        self.dataset.finalize()

    def clear_episode_buffer(self) -> None:
        """Clear the current episode buffer without saving."""
        self.dataset.clear_episode_buffer()
