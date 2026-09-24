"""Choose the yaw sent with a two-dimensional Nav2 goal."""

from collections import deque
import math

DEFAULT_STANDOFF_DISTANCE = 0.6
DEFAULT_OBJECT_CLEARANCE = 0.35
NAV2_INSCRIBED_OBSTACLE_COST = 253

_APPROACH_ANGLE_OFFSETS = tuple(
    math.radians(degrees)
    for degrees in (0, 30, -30, 60, -60, 90, -90, 120, -120, 180)
)


def calculate_detection_standoff(
    size_x: float = 0.0,
    size_y: float = 0.0,
    minimum_distance: float = DEFAULT_STANDOFF_DISTANCE,
    clearance: float = DEFAULT_OBJECT_CLEARANCE,
) -> float:
    """Return center-to-goal distance using object radius plus clearance."""
    dimensions = []
    for value in (size_x, size_y):
        try:
            dimension = abs(float(value))
        except (TypeError, ValueError):
            continue
        if math.isfinite(dimension):
            dimensions.append(dimension)
    object_radius = max(dimensions, default=0.0) * 0.5
    return max(float(minimum_distance), object_radius + float(clearance))


def generate_standoff_candidates(
    confirmed_pose: dict,
    target_x: float,
    target_y: float,
    standoff_distance: float = DEFAULT_STANDOFF_DISTANCE,
) -> tuple[list[tuple[float, float]], float]:
    """Generate near-side and alternate points on a ring around a target."""
    if not math.isfinite(standoff_distance) or standoff_distance < 0.0:
        raise ValueError("standoff_distance must be a non-negative finite number")

    robot_x = float(confirmed_pose["x"])
    robot_y = float(confirmed_pose["y"])
    target_x = float(target_x)
    target_y = float(target_y)
    target_distance = math.hypot(target_x - robot_x, target_y - robot_y)
    if target_distance <= standoff_distance or target_distance == 0.0:
        return [(robot_x, robot_y)], target_distance

    angle_to_robot = math.atan2(robot_y - target_y, robot_x - target_x)
    candidates = [
        (
            target_x + math.cos(angle_to_robot + offset) * standoff_distance,
            target_y + math.sin(angle_to_robot + offset) * standoff_distance,
        )
        for offset in _APPROACH_ANGLE_OFFSETS
    ]
    return candidates, target_distance


def normalize_costmap(costmap_message) -> dict:
    """Convert a nav2_msgs/Costmap-like object into a calculation-only grid."""
    metadata = getattr(costmap_message, "metadata", None)
    if metadata is None:
        raise ValueError("costmap response is missing metadata")
    resolution = float(getattr(metadata, "resolution", 0.0))
    size_x = int(getattr(metadata, "size_x", 0))
    size_y = int(getattr(metadata, "size_y", 0))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("costmap resolution must be positive")
    if size_x <= 0 or size_y <= 0:
        raise ValueError("costmap dimensions must be positive")

    origin = getattr(metadata, "origin", None)
    position = getattr(origin, "position", None)
    orientation = getattr(origin, "orientation", None)
    origin_x = float(getattr(position, "x", 0.0))
    origin_y = float(getattr(position, "y", 0.0))
    qx = float(getattr(orientation, "x", 0.0))
    qy = float(getattr(orientation, "y", 0.0))
    qz = float(getattr(orientation, "z", 0.0))
    qw = float(getattr(orientation, "w", 1.0))
    if not all(math.isfinite(value) for value in (origin_x, origin_y, qx, qy, qz, qw)):
        raise ValueError("costmap origin contains a non-finite value")
    origin_yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    data = getattr(costmap_message, "data", None)
    if data is None or len(data) < size_x * size_y:
        raise ValueError("costmap data is incomplete")
    return {
        "resolution": resolution,
        "size_x": size_x,
        "size_y": size_y,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "origin_yaw": origin_yaw,
        "data": data,
    }


def _costmap_cell_at(costmap: dict, x: float, y: float) -> tuple[int, int] | None:
    if not all(math.isfinite(float(value)) for value in (x, y)):
        return None
    dx = float(x) - float(costmap["origin_x"])
    dy = float(y) - float(costmap["origin_y"])
    yaw = float(costmap.get("origin_yaw", 0.0))
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    resolution = float(costmap["resolution"])
    cell_x = math.floor(local_x / resolution)
    cell_y = math.floor(local_y / resolution)
    size_x = int(costmap["size_x"])
    size_y = int(costmap["size_y"])
    if not (0 <= cell_x < size_x and 0 <= cell_y < size_y):
        return None
    return cell_x, cell_y


def _costmap_cell_cost(costmap: dict, cell_x: int, cell_y: int) -> int:
    size_x = int(costmap["size_x"])
    return int(costmap["data"][cell_y * size_x + cell_x])


def _is_traversable_cost(cost: int) -> bool:
    """Reject lethal, inscribed, and unknown costmap values."""
    return 0 <= int(cost) < NAV2_INSCRIBED_OBSTACLE_COST


def costmap_cost_at(costmap: dict, x: float, y: float) -> int | None:
    """Return one world-coordinate cost, or None when outside the costmap."""
    cell = _costmap_cell_at(costmap, x, y)
    if cell is None:
        return None
    return _costmap_cell_cost(costmap, *cell)


def _reachable_candidate_steps(
    costmap: dict,
    start: tuple[float, float],
    candidate_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    start_cell = _costmap_cell_at(costmap, *start)
    if (
        start_cell is None
        or not _is_traversable_cost(_costmap_cell_cost(costmap, *start_cell))
    ):
        return {}

    size_x = int(costmap["size_x"])
    size_y = int(costmap["size_y"])
    visited = bytearray(size_x * size_y)
    queue = deque([(start_cell[0], start_cell[1], 0)])
    visited[start_cell[1] * size_x + start_cell[0]] = 1
    remaining = set(candidate_cells)
    reached = {}
    neighbors = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    while queue and remaining:
        cell_x, cell_y, steps = queue.popleft()
        cell = (cell_x, cell_y)
        if cell in remaining:
            reached[cell] = steps
            remaining.remove(cell)
        for offset_x, offset_y in neighbors:
            next_x = cell_x + offset_x
            next_y = cell_y + offset_y
            if not (0 <= next_x < size_x and 0 <= next_y < size_y):
                continue
            index = next_y * size_x + next_x
            if visited[index]:
                continue
            if not _is_traversable_cost(
                _costmap_cell_cost(costmap, next_x, next_y)
            ):
                continue
            if offset_x and offset_y:
                horizontal_blocked = (
                    not _is_traversable_cost(
                        _costmap_cell_cost(costmap, cell_x + offset_x, cell_y)
                    )
                )
                vertical_blocked = (
                    not _is_traversable_cost(
                        _costmap_cell_cost(costmap, cell_x, cell_y + offset_y)
                    )
                )
                if horizontal_blocked or vertical_blocked:
                    continue
            visited[index] = 1
            queue.append((next_x, next_y, steps + 1))
    return reached


def select_costmap_candidate(
    candidates: list[tuple[float, float]],
    costmap: dict,
    start: tuple[float, float] | None = None,
) -> tuple[float, float, int] | None:
    """Select a free, connected candidate with a near-side preference."""
    valid = []
    for index, (x, y) in enumerate(candidates):
        cell = _costmap_cell_at(costmap, x, y)
        cost = costmap_cost_at(costmap, x, y)
        if (
            cell is None
            or cost is None
            or not _is_traversable_cost(cost)
        ):
            continue
        valid.append((index, x, y, cost, cell))
    if not valid:
        return None

    if start is None:
        _, x, y, cost, _ = min(
            valid,
            key=lambda item: (item[3] + item[0] * 16, item[0]),
        )
        return x, y, cost

    reachable_steps = _reachable_candidate_steps(
        costmap,
        start,
        {item[4] for item in valid},
    )
    reachable = [item for item in valid if item[4] in reachable_steps]
    if not reachable:
        return None
    resolution = float(costmap["resolution"])
    _, x, y, cost, cell = min(
        reachable,
        key=lambda item: (
            reachable_steps[item[4]] * resolution
            + item[0] * 0.15
            + item[3] / 504.0,
            item[0],
        ),
    )
    return x, y, cost


def calculate_standoff_goal(
    confirmed_pose: dict,
    target_x: float,
    target_y: float,
    standoff_distance: float = DEFAULT_STANDOFF_DISTANCE,
) -> tuple[float, float, float]:
    """Return a reachable point before an obstacle-centered detection target."""
    candidates, target_distance = generate_standoff_candidates(
        confirmed_pose,
        target_x,
        target_y,
        standoff_distance,
    )
    return candidates[0][0], candidates[0][1], target_distance


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
