#!/usr/bin/env python3
"""
Example: Load custom 3D objects into the simulation.

This script shows how to load your own downloaded 3D meshes (OBJ, STL)
into the robosuite simulation and record synthetic data with them.

SUPPORTED FORMATS:
- .obj (Wavefront OBJ)
- .stl (STL mesh)

WHERE TO GET 3D MODELS:
- Objaverse: https://objaverse.allenai.org/
- Sketchfab: https://sketchfab.com/
- Thingiverse: https://www.thingiverse.com/
- TurboSquid: https://www.turbosquid.com/
- Free3D: https://free3d.com/

USAGE:
    1. Download an OBJ or STL file
    2. Configure the object in a YAML file or directly in code
    3. Run with --custom-objects-config or programmatically

EXAMPLE - Direct Python usage:
    
    from franka_pipeline.sim.custom_objects_env import (
        CustomObjectsSimEnv, 
        CustomObjectConfig
    )
    
    # Define your objects
    objects = [
        CustomObjectConfig(
            name="mug",
            mesh_path="/path/to/mug.obj",
            position=(0.0, 0.0, 0.02),    # x, y, z on table
            rotation=(0, 0, 45),           # roll, pitch, yaw in degrees
            scale=0.1,                     # scale factor
            rgba=(0.8, 0.2, 0.2, 1.0),    # red color
        ),
        CustomObjectConfig(
            name="bottle",
            mesh_path="/path/to/bottle.stl",
            position=(0.15, -0.1, 0.02),
            scale=0.05,
            rgba=(0.2, 0.6, 0.2, 1.0),    # green color
        ),
    ]
    
    # Create environment
    sim_env = CustomObjectsSimEnv(
        camera_width=640,
        camera_height=480,
        custom_objects=objects,
    )
    
    # Now use sim_env just like RobosuiteSimEnv
    obs = sim_env.reset()
    
    # Get object positions
    positions = sim_env.get_object_positions()
    print(f"Mug position: {positions['mug']}")

EXAMPLE - YAML config file (objects.yaml):

    objects:
      - name: coffee_mug
        mesh_path: /path/to/models/mug.obj
        position: [0.0, 0.0, 0.02]
        rotation: [0, 0, 45]
        scale: 0.1
        rgba: [0.8, 0.3, 0.1, 1.0]
        
      - name: bowl
        mesh_path: /path/to/models/bowl.obj
        position: [0.15, 0.1, 0.01]
        scale: 0.08
        # No rotation = upright
        # No rgba = gray default

Then load with:
    
    import yaml
    from franka_pipeline.sim.custom_objects_env import (
        CustomObjectsSimEnv,
        CustomObjectConfig
    )
    
    with open("objects.yaml") as f:
        config = yaml.safe_load(f)
    
    objects = [CustomObjectConfig(**obj) for obj in config["objects"]]
    sim_env = CustomObjectsSimEnv(custom_objects=objects)

TIPS:
- Scale your objects appropriately (typical tabletop objects: 0.01 - 0.2)
- Position z=0 is the table surface, use small positive values (0.01-0.05)
- Position (0, 0) is roughly center of table in front of robot
- X is forward/backward, Y is left/right from robot's perspective
- Download objects as OBJ with materials for better visuals
"""

import argparse
import yaml
from pathlib import Path


def load_objects_from_yaml(yaml_path: str) -> list:
    """Load object configurations from a YAML file."""
    from franka_pipeline.sim.custom_objects_env import CustomObjectConfig
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    objects = []
    for obj_config in config.get("objects", []):
        # Convert lists to tuples for dataclass
        if "position" in obj_config and isinstance(obj_config["position"], list):
            obj_config["position"] = tuple(obj_config["position"])
        if "rotation" in obj_config and isinstance(obj_config["rotation"], list):
            obj_config["rotation"] = tuple(obj_config["rotation"])
        if "rgba" in obj_config and isinstance(obj_config["rgba"], list):
            obj_config["rgba"] = tuple(obj_config["rgba"])
        
        objects.append(CustomObjectConfig(**obj_config))
    
    return objects


def main():
    parser = argparse.ArgumentParser(description="Test custom objects in simulation")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML config file with object definitions",
    )
    parser.add_argument(
        "--mesh", "-m",
        type=str,
        help="Path to a single mesh file to test",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Scale for test mesh (default: 0.1)",
    )
    args = parser.parse_args()
    
    from franka_pipeline.sim.custom_objects_env import (
        CustomObjectsSimEnv,
        CustomObjectConfig,
    )
    
    # Load objects
    if args.config:
        objects = load_objects_from_yaml(args.config)
        print(f"Loaded {len(objects)} objects from {args.config}")
    elif args.mesh:
        mesh_path = Path(args.mesh)
        objects = [
            CustomObjectConfig(
                name=mesh_path.stem,
                mesh_path=str(mesh_path),
                position=(0.0, 0.0, 0.02),
                scale=args.scale,
            )
        ]
        print(f"Loading single mesh: {mesh_path.name}")
    else:
        print(__doc__)
        print("\nNo objects specified. Use --config or --mesh to load objects.")
        return
    
    # Create environment
    print("Creating environment...")
    sim_env = CustomObjectsSimEnv(
        camera_width=640,
        camera_height=480,
        custom_objects=objects,
    )
    
    print("Environment created! Objects:")
    positions = sim_env.get_object_positions()
    for name, pos in positions.items():
        print(f"  {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    
    print("\nPress Ctrl+C to exit...")
    try:
        while True:
            sim_env.step(np.zeros(7))  # Zero action
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sim_env.close()


if __name__ == "__main__":
    import numpy as np
    main()
