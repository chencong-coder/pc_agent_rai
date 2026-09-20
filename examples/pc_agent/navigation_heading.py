"""Choose the yaw sent with a two-dimensional Nav2 goal."""

import math


def resolve_navigation_yaw(
    confirmed_pose: dict,
    target_x: float,
    target_y: float,
) -> tuple[float, str]:
    """Prefer RViz's selected yaw, otherwise face from the robot to the goal."""
    initial_pose_yaw = confirmed_pose.get("initial_pose_yaw")
    if initial_pose_yaw is not None:
        try:
            selected_yaw = float(initial_pose_yaw)
        except (TypeError, ValueError):
            selected_yaw = math.nan
        if math.isfinite(selected_yaw):
            return selected_yaw, "initialpose"

    robot_x = float(confirmed_pose["x"])
    robot_y = float(confirmed_pose["y"])
    return math.atan2(target_y - robot_y, target_x - robot_x), "calculated"
