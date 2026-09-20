import unittest
from types import SimpleNamespace

from .detection_selection import (
    is_detection_navigation_request,
    select_detection_targets,
)


def _detection(class_name: str, direction: str):
    return SimpleNamespace(class_name=class_name, direction=direction)


class DetectionSelectionTests(unittest.TestCase):
    def test_recognizes_snapshot_navigation_commands(self):
        self.assertTrue(is_detection_navigation_request("去前方的椅子"))
        self.assertTrue(is_detection_navigation_request("导航到第二个目标"))
        self.assertFalse(is_detection_navigation_request("周围有什么"))
        self.assertFalse(
            is_detection_navigation_request("去 map 坐标 x=-1.0, y=2.0")
        )

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

    def test_selects_numbered_target(self):
        detections = [
            _detection("chair", "正前方"),
            _detection("cabinet", "左侧"),
        ]

        selected = select_detection_targets(detections, "去第二个目标")

        self.assertEqual(selected, [detections[1]])


if __name__ == "__main__":
    unittest.main()
