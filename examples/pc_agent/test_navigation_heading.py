import math
import unittest

from .navigation_heading import resolve_navigation_yaw


class NavigationHeadingTests(unittest.TestCase):
    def test_uses_rviz_initial_pose_yaw_when_available(self):
        yaw, source = resolve_navigation_yaw(
            {"x": 1.0, "y": 2.0, "initial_pose_yaw": 1.25},
            target_x=10.0,
            target_y=20.0,
        )

        self.assertAlmostEqual(yaw, 1.25)
        self.assertEqual(source, "initialpose")

    def test_calculates_yaw_when_initial_pose_was_not_used(self):
        yaw, source = resolve_navigation_yaw(
            {"x": 1.0, "y": 2.0, "initial_pose_yaw": None},
            target_x=1.0,
            target_y=4.0,
        )

        self.assertAlmostEqual(yaw, math.pi / 2)
        self.assertEqual(source, "calculated")

    def test_invalid_initial_pose_yaw_falls_back_to_calculation(self):
        yaw, source = resolve_navigation_yaw(
            {"x": 0.0, "y": 0.0, "initial_pose_yaw": math.nan},
            target_x=1.0,
            target_y=0.0,
        )

        self.assertAlmostEqual(yaw, 0.0)
        self.assertEqual(source, "calculated")


if __name__ == "__main__":
    unittest.main()
