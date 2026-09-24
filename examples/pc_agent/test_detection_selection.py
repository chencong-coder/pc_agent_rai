import unittest
from types import SimpleNamespace

from .detection_selection import (
    direction_from_robot_frame,
    is_detection_navigation_request,
    select_one_to_one_matches,
    select_detection_targets,
    summarize_detection_directions,
)


def _detection(class_name: str, direction: str):
    return SimpleNamespace(class_name=class_name, direction=direction)


class DetectionSelectionTests(unittest.TestCase):
    def test_track_matching_is_one_to_one_per_frame(self):
        matches = select_one_to_one_matches([
            (0.1, 0, 0),
            (0.2, 0, 1),
            (0.3, 1, 0),
            (0.4, 1, 1),
        ])

        self.assertEqual(matches, [(0.1, 0, 0), (0.4, 1, 1)])

    def test_splits_slightly_left_and_right_front_targets(self):
        self.assertEqual(direction_from_robot_frame(2.0, 0.35), "前方偏左")
        self.assertEqual(direction_from_robot_frame(2.0, -0.35), "前方偏右")
        self.assertEqual(direction_from_robot_frame(2.0, 0.05), "正前方")
        self.assertEqual(direction_from_robot_frame(float("nan"), 0.0), "方向未知")

    def test_recognizes_snapshot_navigation_commands(self):
        self.assertTrue(is_detection_navigation_request("去前方的椅子"))
        self.assertTrue(is_detection_navigation_request("导航到第二个目标"))
        self.assertFalse(is_detection_navigation_request("周围有什么"))
        self.assertFalse(
            is_detection_navigation_request("去 map 坐标 x=-1.0, y=2.0")
        )
        self.assertTrue(is_detection_navigation_request("去刚才那个目标"))

    def test_selects_chinese_class_and_direction_together(self):
        detections = [
            _detection("chair", "正前方"),
            _detection("chair", "左侧"),
            _detection("cabinet", "正前方"),
        ]

        selected = select_detection_targets(detections, "去前方的椅子")

        self.assertEqual(selected, [detections[0]])

    def test_keeps_multiple_snapshot_matches_ambiguous(self):
        detections = [
            _detection("chair", "正前方"),
            _detection("chair", "正前方"),
        ]

        selected = select_detection_targets(detections, "去前方的椅子")

        self.assertEqual(selected, detections)

    def test_selects_front_offset_without_an_ordinal(self):
        detections = [
            _detection("chair", "前方偏左"),
            _detection("chair", "前方偏右"),
        ]

        selected = select_detection_targets(detections, "去前方偏右的椅子")

        self.assertEqual(selected, [detections[1]])

    def test_accepts_common_left_right_front_phrasing(self):
        detections = [
            _detection("chair", "前方偏左"),
            _detection("chair", "前方偏右"),
        ]

        self.assertEqual(
            select_detection_targets(detections, "去前方左侧的椅子"),
            [detections[0]],
        )

    def test_straight_ahead_query_includes_both_front_offsets(self):
        detections = [
            _detection("chair", "前方偏左"),
            _detection("chair", "前方偏右"),
        ]

        selected = select_detection_targets(detections, "去正前方的椅子")

        self.assertEqual(selected, detections)

    def test_selects_numbered_target(self):
        detections = [
            _detection("chair", "正前方"),
            _detection("cabinet", "左侧"),
        ]

        selected = select_detection_targets(detections, "去第二个目标")

        self.assertEqual(selected, [detections[1]])

    def test_generic_target_uses_only_the_latest_snapshot(self):
        detections = [
            _detection("chair", "前方偏左"),
            _detection("table", "右侧"),
        ]

        self.assertEqual(
            select_detection_targets(detections, "去刚才那个目标"),
            detections,
        )

    def test_summarizes_counts_by_exact_direction_and_class(self):
        detections = [
            _detection("chair", "左前方"),
            _detection("cabinet", "左前方"),
            _detection("chair", "右侧"),
            _detection("chair", "左前方"),
        ]

        summary = summarize_detection_directions(detections)

        self.assertEqual(
            summary,
            "小车左前方有2把椅子、1个柜子；小车右侧有1把椅子",
        )


if __name__ == "__main__":
    unittest.main()
