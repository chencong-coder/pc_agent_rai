"""Deterministic selection of targets from the latest detection snapshot."""

import math
import re


CLASS_NAMES_ZH = {
    "bed": "床",
    "chair": "椅子",
    "sofa": "沙发",
    "table": "桌子",
    "desk": "书桌",
    "cabinet": "柜子",
    "door": "门",
    "window": "窗户",
    "bookshelf": "书架",
    "toilet": "马桶",
    "sink": "水槽",
}

CLASS_COUNTERS_ZH = {
    "bed": "张",
    "chair": "把",
    "sofa": "个",
    "table": "张",
    "desk": "张",
    "cabinet": "个",
    "door": "扇",
    "window": "扇",
    "bookshelf": "个",
    "toilet": "个",
    "sink": "个",
}

DIRECTION_ORDER = (
    "正前方",
    "前方偏左",
    "左前方",
    "左侧",
    "左后方",
    "正后方",
    "右后方",
    "右侧",
    "右前方",
    "前方偏右",
    "方向未知",
)

DIRECTION_ALIASES = {
    "正前方偏左": "前方偏左",
    "正前方偏右": "前方偏右",
    "前方偏左": "前方偏左",
    "前方偏右": "前方偏右",
    "偏左前方": "前方偏左",
    "偏右前方": "前方偏右",
    "左前面": "前方偏左",
    "右前面": "前方偏右",
    "前方左侧": "前方偏左",
    "前方右侧": "前方偏右",
    "左前": "前方偏左",
    "右前": "前方偏右",
    "偏左": "前方偏左",
    "偏右": "前方偏右",
    "正前方": "正前方",
    "左前方": "左前方",
    "右前方": "右前方",
    "正后方": "正后方",
    "左后方": "左后方",
    "右后方": "右后方",
    "左侧": "左侧",
    "右侧": "右侧",
    "前方": "正前方",
    "前面": "正前方",
    "后方": "正后方",
    "后面": "正后方",
    "左边": "左侧",
    "左面": "左侧",
    "右边": "右侧",
    "右面": "右侧",
}

_NUMBER_WORDS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}


def select_one_to_one_matches(candidates: list[tuple]) -> list[tuple]:
    """Greedily choose nearest matches with one track and detection per frame."""
    selected = []
    matched_tracks = set()
    matched_detections = set()
    for candidate in sorted(candidates):
        _, track_index, detection_index = candidate
        if (
            track_index in matched_tracks
            or detection_index in matched_detections
        ):
            continue
        matched_tracks.add(track_index)
        matched_detections.add(detection_index)
        selected.append(candidate)
    return selected


def direction_from_robot_frame(x: float, y: float) -> str:
    """Describe a target direction using base_link's x-forward convention."""
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return "方向未知"
    if not math.isfinite(x) or not math.isfinite(y):
        return "方向未知"
    angle = math.atan2(y, x)
    center_tolerance = math.radians(5.0)
    if -center_tolerance <= angle <= center_tolerance:
        return "正前方"
    if center_tolerance < angle < math.pi / 8:
        return "前方偏左"
    if math.pi / 8 <= angle < 3 * math.pi / 8:
        return "左前方"
    if 3 * math.pi / 8 <= angle < 5 * math.pi / 8:
        return "左侧"
    if 5 * math.pi / 8 <= angle < 7 * math.pi / 8:
        return "左后方"
    if angle >= 7 * math.pi / 8 or angle < -7 * math.pi / 8:
        return "正后方"
    if -7 * math.pi / 8 <= angle < -5 * math.pi / 8:
        return "右后方"
    if -5 * math.pi / 8 <= angle < -3 * math.pi / 8:
        return "右侧"
    if -math.pi / 8 < angle < -center_tolerance:
        return "前方偏右"
    return "右前方"


def is_detection_navigation_request(prompt: str) -> bool:
    """Return whether a command should navigate using the saved snapshot."""
    query = str(prompt or "").strip().lower()
    if re.search(
        r"(?:\bx\b|x坐标|横坐标)\s*[=:：].*(?:\by\b|y坐标|纵坐标)\s*[=:：]",
        query,
    ):
        return False
    navigation_words = ("去", "找", "导航到", "前往", "过去", "带我到", "靠近")
    if not any(word in query for word in navigation_words):
        return False
    if any(word in query for word in ("停下", "停止", "取消")):
        return False

    class_words = tuple(CLASS_NAMES_ZH) + tuple(CLASS_NAMES_ZH.values())
    has_numbered_target = bool(
        re.search(r"第\s*(?:\d+|一|二|三|四|五)\s*(?:个|号)?", query)
    )
    has_generic_target = any(
        word in query for word in ("目标", "物体", "东西", "那里", "刚才")
    )
    return (
        any(word in query for word in class_words)
        or any(word in query for word in DIRECTION_ALIASES)
        or has_numbered_target
        or has_generic_target
    )


def summarize_detection_directions(detections: list) -> str:
    """Summarize exact object counts grouped by robot-relative direction."""
    grouped: dict[str, dict[str, int]] = {}
    for detection in detections:
        direction = str(detection.direction or "方向未知")
        class_name = str(detection.class_name).lower()
        class_counts = grouped.setdefault(direction, {})
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    ordered_directions = [
        direction for direction in DIRECTION_ORDER if direction in grouped
    ]
    ordered_directions.extend(
        direction for direction in grouped if direction not in DIRECTION_ORDER
    )

    summaries = []
    for direction in ordered_directions:
        object_counts = []
        for class_name, count in grouped[direction].items():
            display_name = CLASS_NAMES_ZH.get(class_name, class_name)
            counter = CLASS_COUNTERS_ZH.get(class_name, "个")
            object_counts.append(f"{count}{counter}{display_name}")
        objects_text = "、".join(object_counts)
        if direction == "方向未知":
            summaries.append(f"方向未知的目标有{objects_text}")
        else:
            summaries.append(f"小车{direction}有{objects_text}")
    return "；".join(summaries)


def select_detection_targets(detections: list, target: str) -> list:
    """Filter one saved snapshot by ordinal, class and relative direction."""
    query = str(target or "").strip().lower()
    index_match = re.search(
        r"第\s*(\d+|一|二|三|四|五)\s*(?:个|号)?",
        query,
    )
    if index_match is None and query.isdigit():
        index_match = re.match(r"(\d+)", query)
    if index_match is not None:
        index_text = index_match.group(1)
        ordinal = (
            int(index_text)
            if index_text.isdigit()
            else _NUMBER_WORDS.get(index_text, 0)
        )
        index = ordinal - 1
        return [detections[index]] if 0 <= index < len(detections) else []

    class_filter = next(
        (
            class_name
            for class_name, chinese_name in CLASS_NAMES_ZH.items()
            if class_name in query or chinese_name in query
        ),
        None,
    )
    direction_filter = next(
        (
            canonical
            for alias, canonical in sorted(
                DIRECTION_ALIASES.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
            if alias in query
        ),
        None,
    )

    selected = list(detections)
    if class_filter is not None:
        selected = [
            detection
            for detection in selected
            if detection.class_name.lower() == class_filter
        ]
    if direction_filter is not None:
        matching_directions = {direction_filter}
        if direction_filter == "正前方":
            matching_directions.update(("前方偏左", "前方偏右"))
        selected = [
            detection
            for detection in selected
            if detection.direction in matching_directions
        ]
    if class_filter is None and direction_filter is None:
        if any(word in query for word in ("目标", "物体", "东西", "那里", "刚才")):
            return list(detections)
        selected = [
            detection
            for detection in detections
            if query in detection.class_name.lower()
            or query in detection.direction.lower()
        ]
    return selected
