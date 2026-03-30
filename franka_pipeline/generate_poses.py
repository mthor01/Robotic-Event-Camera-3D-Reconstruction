#!/usr/bin/env python3
"""Generate calibration/recording poses and save to .npy file.

Supports RandomSphereAgent and RandomHemisphereAgent with full
customisation of target, center, radius, and safety parameters.

Usage:
    python generate_poses.py -o calibration_poses.npy
    python generate_poses.py --target 0.4 0.0 0.2 --radius 0.25 --num-poses 20
    python generate_poses.py --agent sphere --center 0.5 0.0 0.4 --target 0.4 0.0 0.2
    python generate_poses.py --seed 42 -o my_poses.npy
"""

import argparse
import numpy as np

from franka_pipeline.agents.random_sphere_agent import RandomSphereAgent
from franka_pipeline.agents.random_hemisphere_agent import RandomHemisphereAgent
import config_defaults as cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate agent poses and save to .npy file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="calibration_poses.npy",
        help="Output .npy file path (default: calibration_poses.npy)",
    )
    parser.add_argument(
        "--agent", "-a",
        type=str,
        default="random_hemisphere",
        choices=["sphere", "random_hemisphere"],
        help="Agent type (default: random_hemisphere)",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[cfg.TARGET_X, cfg.TARGET_Y, cfg.TARGET_Z],
        metavar=("X", "Y", "Z"),
        help=f"Target point the camera looks at (default: {cfg.TARGET_X} {cfg.TARGET_Y} {cfg.TARGET_Z})",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Sphere center (default: same as target for hemisphere, or config for sphere)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=cfg.SPHERE_RADIUS,
        help=f"Sphere radius (default: {cfg.SPHERE_RADIUS})",
    )
    parser.add_argument(
        "--inner-radius",
        type=float,
        default=cfg.INNER_RADIUS,
        help=f"Inner exclusion radius for hemisphere (default: {cfg.INNER_RADIUS})",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=cfg.NUM_POSES,
        help=f"Number of poses (default: {cfg.NUM_POSES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.RANDOM_SEED,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--base-exclusion-radius",
        type=float,
        default=cfg.BASE_EXCLUSION_RADIUS,
        help=f"Min xy distance from robot base (default: {cfg.BASE_EXCLUSION_RADIUS})",
    )
    parser.add_argument(
        "--base-max-radius",
        type=float,
        default=cfg.BASE_MAX_RADIUS,
        help=f"Max xy distance from robot base (default: {cfg.BASE_MAX_RADIUS})",
    )
    parser.add_argument(
        "--min-z",
        type=float,
        default=cfg.MIN_Z_HEIGHT,
        help=f"Minimum z-height (default: {cfg.MIN_Z_HEIGHT})",
    )
    parser.add_argument(
        "--lock-rotation",
        action=argparse.BooleanOptionalAction,
        default=cfg.LOCK_ROTATION_HORIZONTAL,
        help=f"Lock Y-axis horizontal (default: {cfg.LOCK_ROTATION_HORIZONTAL})",
    )

    args = parser.parse_args()
    target = np.array(args.target)

    if args.agent == "random_hemisphere":
        center = target  # hemisphere always centered on target
        agent = RandomHemisphereAgent(
            center=center,
            radius=args.radius,
            inner_radius=args.inner_radius,
            num_poses=args.num_poses,
            wait_time=0.0,
            seed=args.seed,
            loop=False,
            target_point=target,
            base_exclusion_radius=args.base_exclusion_radius,
            base_max_radius=args.base_max_radius,
            min_z_height=args.min_z,
            lock_rotation_horizontal=args.lock_rotation,
        )
    else:
        center = np.array(args.center) if args.center else np.array(
            [cfg.SPHERE_CENTER_X, cfg.SPHERE_CENTER_Y, cfg.SPHERE_CENTER_Z]
        )
        agent = RandomSphereAgent(
            center=center,
            radius=args.radius,
            num_poses=args.num_poses,
            wait_time=0.0,
            seed=args.seed,
            loop=False,
            target_point=target,
        )

    poses = np.asarray(agent.poses, dtype=np.float64)
    np.save(args.output, poses)

    print(f"Agent:       {args.agent}")
    print(f"Target:      {target}")
    print(f"Center:      {center}")
    print(f"Radius:      {args.radius}")
    print(f"Poses:       {len(poses)}")
    print(f"Saved to:    {args.output}")
    print()
    np.set_printoptions(precision=4, suppress=True)
    for i, p in enumerate(poses):
        print(f"  {i:2d}: {p}")


if __name__ == "__main__":
    main()
