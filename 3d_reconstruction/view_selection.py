"""Shared pose-based source-view selection utilities."""

from __future__ import annotations

import numpy as np


def pose_neighbours(
    camera_centers: np.ndarray,
    target_idx: int,
    direction: int,
    count: int,
    move_threshold: float,
) -> list[int]:
    """Return geometrically spaced neighbours in one temporal direction."""
    neighbours: list[int] = []
    anchor = target_idx
    cursor = target_idx + direction
    while 0 <= cursor < len(camera_centers) and len(neighbours) < count:
        moved = np.linalg.norm(camera_centers[cursor] - camera_centers[anchor])
        if moved >= move_threshold:
            neighbours.append(cursor)
            anchor = cursor
        cursor += direction
    return neighbours


def select_pose_views(
    camera_centers: np.ndarray,
    target_idx: int,
    num_views: int,
    move_threshold: float,
    allow_unbalanced: bool = True,
) -> list[int] | None:
    """Select a fixed-size target/source tuple from camera-center motion.

    The default keeps a fixed target + past slots + future slots layout. With
    ``allow_unbalanced``, unavailable boundary slots are represented by ``-1``
    instead of being filled from the opposite temporal side. This includes a
    reference-only tuple when no qualifying source exists. ``None`` is returned
    only if strict balancing is requested and a slot is missing.
    """
    if num_views < 1 or num_views % 2 != 1:
        raise ValueError("pose-based selection requires a positive odd num_views")
    if num_views == 1:
        return [target_idx]

    source_count = num_views - 1
    per_direction = source_count // 2
    before = pose_neighbours(
        camera_centers, target_idx, -1, per_direction, move_threshold
    )
    after = pose_neighbours(
        camera_centers, target_idx, 1, per_direction, move_threshold
    )

    if not allow_unbalanced:
        if len(before) < per_direction or len(after) < per_direction:
            return None
        return [target_idx, *before[:per_direction], *after[:per_direction]]

    before_slots = [*before, *([-1] * (per_direction - len(before)))]
    after_slots = [*after, *([-1] * (per_direction - len(after)))]
    return [target_idx, *before_slots, *after_slots]


def build_pose_view_ids(
    camera_centers: np.ndarray,
    num_views: int,
    move_threshold: float,
    allow_unbalanced: bool = True,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Build source tuples and valid target indices for a complete sequence."""
    view_ids: dict[int, list[int]] = {}
    valid: list[int] = []
    for target_idx in range(len(camera_centers)):
        selected = select_pose_views(
            camera_centers,
            target_idx,
            num_views,
            move_threshold,
            allow_unbalanced,
        )
        if selected is None:
            continue
        view_ids[target_idx] = selected
        valid.append(target_idx)
    return view_ids, np.asarray(valid, dtype=np.int64)


def pose_layout_counts(view_ids: dict[int, list[int]]) -> dict[str, int]:
    """Count balanced, asymmetric, and fully one-sided target tuples."""
    counts = {
        "balanced": 0,
        "asymmetric": 0,
        "one_sided": 0,
        "reference_only": 0,
    }
    for target_idx, selected in view_ids.items():
        before = sum(0 <= source_idx < target_idx for source_idx in selected[1:])
        after = sum(source_idx > target_idx for source_idx in selected[1:])
        if before == 0 and after == 0:
            counts["reference_only"] += 1
        elif before == after:
            counts["balanced"] += 1
        elif before == 0 or after == 0:
            counts["one_sided"] += 1
        else:
            counts["asymmetric"] += 1
    return counts
