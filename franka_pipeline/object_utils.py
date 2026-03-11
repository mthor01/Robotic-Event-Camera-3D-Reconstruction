"""
Object definitions and utilities for synthetic data generation.

This module centralises all object-related logic so that the main pipeline
script (my_main.py) does not need to know about individual object definitions,
sizes, YAML loading, or spawn-position calculations.

Public API
----------
AVAILABLE_OBJECTS          – list of all objects available for multi-object recording
DEFAULT_OBJECT_POSITION    – default world-frame spawn position derived from config

get_object_height(obj)     – return the height of an object in metres
create_object_config(d)    – build an ObjectConfig from a dict entry
list_available_objects()   – pretty-print all available objects

load_custom_objects_from_yaml(path)  – load legacy CustomObjectConfig list from YAML
load_objects_from_yaml(path)         – load new-format ObjectConfig list from YAML
has_type_field(path)                 – detect which YAML format is in use
"""

import yaml
import numpy as np

import config_defaults as cfg
from franka_pipeline.sim.custom_objects_env import CustomObjectConfig
from franka_pipeline.sim.empty_table_env import ObjectConfig

# ============================================================================
# Available objects for multi-object synthetic data generation
# ============================================================================

AVAILABLE_OBJECTS = [
    # Primitive shapes (different sizes)
    {"name": "cube_small",  "type": "cube",     "size": 0.015,         "rgba": (1.0, 0.0, 0.0, 1.0)},
    {"name": "cube_medium", "type": "cube",     "size": 0.025,         "rgba": (1.0, 0.0, 0.0, 1.0)},
    {"name": "cube_large",  "type": "cube",     "size": 0.035,         "rgba": (1.0, 0.0, 0.0, 1.0)},

    {"name": "sphere_small",  "type": "sphere", "size": [0.015],       "rgba": (0.0, 1.0, 0.0, 1.0)},
    {"name": "sphere_medium", "type": "sphere", "size": [0.025],       "rgba": (0.0, 1.0, 0.0, 1.0)},
    {"name": "sphere_large",  "type": "sphere", "size": [0.04],        "rgba": (0.0, 1.0, 0.0, 1.0)},

    {"name": "cylinder_small",  "type": "cylinder", "size": [0.015, 0.02], "rgba": (0.0, 0.0, 1.0, 1.0)},
    {"name": "cylinder_medium", "type": "cylinder", "size": [0.02, 0.03],  "rgba": (0.0, 0.0, 1.0, 1.0)},
    {"name": "cylinder_tall",   "type": "cylinder", "size": [0.015, 0.05], "rgba": (0.0, 0.0, 1.0, 1.0)},

    {"name": "capsule_small",  "type": "capsule", "size": [0.012, 0.025], "rgba": (1.0, 1.0, 0.0, 1.0)},
    {"name": "capsule_medium", "type": "capsule", "size": [0.015, 0.03],  "rgba": (1.0, 1.0, 0.0, 1.0)},
    {"name": "capsule_large",  "type": "capsule", "size": [0.02, 0.04],   "rgba": (1.0, 1.0, 0.0, 1.0)},

    # Pre-made XML objects from robosuite
    {"name": "milk",       "type": "milk"},
    {"name": "bread",      "type": "bread"},
    {"name": "cereal",     "type": "cereal"},
    {"name": "can",        "type": "can"},
    {"name": "bottle",     "type": "bottle"},
    {"name": "lemon",      "type": "lemon"},
    {"name": "square_nut", "type": "square_nut"},
    {"name": "round_nut",  "type": "round_nut"},

    # Composite objects from robosuite
    {"name": "pot",             "type": "pot"},
    {"name": "hammer",          "type": "hammer"},
    {"name": "hollow_cylinder", "type": "hollow_cylinder"},
    {"name": "cone",            "type": "cone"},
]

# ============================================================================
# Spawn position helpers
# ============================================================================

# Robot base position in world frame (standard robosuite table setup)
ROBOT_BASE_POS_WORLD = np.array([-0.55, 0.0, 0.9])
TABLE_HEIGHT = 0.82


def get_object_spawn_position() -> tuple:
    """Calculate object spawn position in world frame from the config target in base frame."""
    world_x = ROBOT_BASE_POS_WORLD[0] + cfg.TARGET_X
    world_y = ROBOT_BASE_POS_WORLD[1] + cfg.TARGET_Y
    world_z = TABLE_HEIGHT  # objects sit on the table surface
    return (world_x, world_y, world_z)


DEFAULT_OBJECT_POSITION: tuple = get_object_spawn_position()

# ============================================================================
# Height lookup table for pre-made XML objects (rough estimates)
# ============================================================================

PREMADE_OBJECT_HEIGHTS: dict[str, float] = {
    "milk":            0.14,   # milk carton  ~14 cm
    "bread":           0.08,   # bread loaf   ~ 8 cm
    "cereal":          0.22,   # cereal box   ~22 cm
    "can":             0.12,   # can          ~12 cm
    "bottle":          0.18,   # bottle       ~18 cm
    "lemon":           0.05,   # lemon        ~ 5 cm diameter
    "square_nut":      0.02,   # square nut   ~ 2 cm
    "round_nut":       0.02,   # round nut    ~ 2 cm
    "pot":             0.10,   # pot          ~10 cm
    "hammer":          0.04,   # hammer handle diameter ~4 cm
    "hollow_cylinder": 0.08,   # hollow cylinder ~8 cm
    "cone":            0.08,   # cone         ~ 8 cm
}

# ============================================================================
# Object height utility
# ============================================================================


def get_object_height(obj) -> float:
    """Return the height of an object in metres.

    Accepts either a plain ``dict`` (from AVAILABLE_OBJECTS) or an
    ``ObjectConfig`` / ``CustomObjectConfig`` dataclass instance.

    Size conventions for primitive shapes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``cube``     – *size* is half-extent  →  height = 2 × size
    * ``sphere``   – *size* = [radius]      →  height = 2 × radius  (diameter)
    * ``cylinder`` – *size* = [radius, half_height] → height = 2 × half_height
    * ``capsule``  – *size* = [radius, half_length] → height = 2×radius + 2×half_length
    """
    if isinstance(obj, dict):
        obj_type = obj.get("type", "")
        size = obj.get("size")
    else:
        obj_type = obj.type if obj.type else ""
        size = obj.size

    if obj_type == "cube":
        if isinstance(size, (int, float)):
            return 2 * size
        if isinstance(size, (list, tuple)) and len(size) >= 1:
            return 2 * size[0]
        return 0.05

    if obj_type in ("sphere", "ball"):
        if isinstance(size, (list, tuple)) and len(size) >= 1:
            return 2 * size[0]
        if isinstance(size, (int, float)):
            return 2 * size
        return 0.05

    if obj_type == "cylinder":
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return 2 * size[1]
        return 0.06

    if obj_type == "capsule":
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            radius, half_length = size[0], size[1]
            return 2 * radius + 2 * half_length
        return 0.08

    return PREMADE_OBJECT_HEIGHTS.get(obj_type, 0.05)


# ============================================================================
# Object config factory
# ============================================================================


def create_object_config(
    obj_dict: dict,
    position: tuple = None,
) -> ObjectConfig:
    """Build an :class:`ObjectConfig` from a dictionary entry.

    If *position* is ``None`` the module-level :data:`DEFAULT_OBJECT_POSITION`
    is used so callers never need to know about coordinate-frame details.
    """
    if position is None:
        position = DEFAULT_OBJECT_POSITION
    return ObjectConfig(
        name=obj_dict["name"],
        type=obj_dict["type"],
        position=position,
        rotation=obj_dict.get("rotation"),
        size=obj_dict.get("size"),
        scale=obj_dict.get("scale", 1.0),
        rgba=obj_dict.get("rgba", (1.0, 0.0, 0.0, 1.0)),
        density=obj_dict.get("density", 1000.0),
        material=obj_dict.get("material"),
    )


# ============================================================================
# Pretty-printer
# ============================================================================


def list_available_objects() -> None:
    """Print all available objects for synthetic data generation."""
    spawn_pos = get_object_spawn_position()
    sep = "=" * 70

    print(f"\n{sep}")
    print("AVAILABLE OBJECTS FOR SYNTHETIC DATA GENERATION")
    print(sep)
    print(f"\nObject spawn position (world frame): {spawn_pos}")
    print(f"  Calculated from target (base frame): ({cfg.TARGET_X}, {cfg.TARGET_Y}, {cfg.TARGET_Z})")

    print("\n--- PRIMITIVE SHAPES ---")
    for obj in AVAILABLE_OBJECTS:
        if obj["type"] in ("cube", "box", "sphere", "ball", "cylinder", "capsule"):
            size_str = f", size={obj['size']}" if "size" in obj else ""
            print(f"  - {obj['name']} (type: {obj['type']}{size_str})")

    print("\n--- PRE-MADE XML OBJECTS (robosuite) ---")
    for obj in AVAILABLE_OBJECTS:
        if obj["type"] in ("milk", "bread", "cereal", "can", "bottle", "lemon", "square_nut", "round_nut"):
            print(f"  - {obj['name']} (type: {obj['type']})")

    print("\n--- COMPOSITE OBJECTS (robosuite) ---")
    for obj in AVAILABLE_OBJECTS:
        if obj["type"] in ("pot", "hammer", "hollow_cylinder", "cone"):
            print(f"  - {obj['name']} (type: {obj['type']})")

    print(f"\n{sep}")
    print(f"TOTAL: {len(AVAILABLE_OBJECTS)} objects available")
    print(f"{sep}\n")


# ============================================================================
# YAML loaders
# ============================================================================


def load_custom_objects_from_yaml(yaml_path: str) -> list[CustomObjectConfig]:
    """Load custom object configurations from a YAML file (legacy format)."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    objects = []
    for obj_cfg in config.get("objects", []):
        for key in ("position", "rotation", "rgba"):
            if key in obj_cfg and isinstance(obj_cfg[key], list):
                obj_cfg[key] = tuple(obj_cfg[key])
        objects.append(CustomObjectConfig(**obj_cfg))
    return objects


def load_objects_from_yaml(yaml_path: str) -> list[ObjectConfig]:
    """Load object configurations from a YAML file (new format with *type* field).

    Expected YAML structure::

        objects:
          - name: my_cube
            type: cube
            position: [0.0, 0.0, 0.82]
            rotation: [0, 0, 45]        # roll, pitch, yaw in degrees (optional)
            size: [0.02, 0.02, 0.02]    # depends on type (optional)
            scale: 1.0                  # for pre-made objects (optional)
            rgba: [1.0, 0.0, 0.0, 1.0] # colour (optional)
            density: 1000.0             # kg/m³ (optional)
            material: "WoodRed"         # texture (optional)
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    objects = []
    for obj_cfg in config.get("objects", []):
        for key in ("position", "rotation", "rgba"):
            if key in obj_cfg and isinstance(obj_cfg[key], list):
                obj_cfg[key] = tuple(obj_cfg[key])
        objects.append(ObjectConfig(**obj_cfg))
    return objects


def has_type_field(yaml_path: str) -> bool:
    """Return ``True`` if the YAML config uses the new format (has a *type* field)."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    objects = config.get("objects", [])
    return bool(objects) and "type" in objects[0]
