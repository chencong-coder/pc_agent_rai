import math
import unittest
from types import SimpleNamespace

from .navigation_heading import (
    DEFAULT_STANDOFF_DISTANCE,
    calculate_detection_standoff,
    calculate_standoff_goal,
    costmap_cost_at,
    generate_standoff_candidates,
    normalize_costmap,
    resolve_navigation_yaw,
    select_costmap_candidate,
)


class NavigationHeadingTests(unittest.TestCase):
    @staticmethod
    def _costmap(width=50, height=50, resolution=0.1):
        return {
            "resolution": resolution,
            "size_x": width,
            "size_y": height,
            "origin_x": -2.5,
            "origin_y": -2.5,
            "origin_yaw": 0.0,
            "data": [0] * (width * height),
        }

    @staticmethod
    def _set_cost(costmap, x, y, cost):
        cell_x = math.floor((x - costmap["origin_x"]) / costmap["resolution"])
        cell_y = math.floor((y - costmap["origin_y"]) / costmap["resolution"])
        index = cell_y * costmap["size_x"] + cell_x
        costmap["data"][index] = cost

    def test_object_size_increases_standoff_for_large_furniture(self):
        self.assertAlmostEqual(calculate_detection_standoff(0.5, 0.5), 0.6)
        self.assertAlmostEqual(calculate_detection_standoff(1.6, 0.8), 1.15)

    def test_candidates_start_on_the_side_nearest_the_robot(self):
        candidates, distance = generate_standoff_candidates(
            {"x": 0.0, "y": 0.0},
            target_x=2.0,
            target_y=0.0,
            standoff_distance=0.6,
        )

        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(candidates[0][0], 1.4)
        self.assertAlmostEqual(candidates[0][1], 0.0)
        self.assertGreater(len(candidates), 3)

    def test_costmap_selection_avoids_an_occupied_near_side_point(self):
        costmap = self._costmap()
        candidates, _ = generate_standoff_candidates(
            {"x": 0.0, "y": 0.0},
            target_x=2.0,
            target_y=0.0,
            standoff_distance=0.6,
        )
        self._set_cost(costmap, *candidates[0], 254)

        selected = select_costmap_candidate(candidates, costmap)

        self.assertIsNotNone(selected)
        self.assertNotAlmostEqual(selected[0], candidates[0][0])
        self.assertEqual(selected[2], 0)

    def test_normalizes_humble_costmap_response_shape(self):
        costmap_message = SimpleNamespace(
            metadata=SimpleNamespace(
                resolution=0.5,
                size_x=2,
                size_y=2,
                origin=SimpleNamespace(
                    position=SimpleNamespace(x=-1.0, y=-2.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
            data=[0, 10, 253, 255],
        )

        costmap = normalize_costmap(costmap_message)

        self.assertEqual(costmap["size_x"], 2)
        self.assertEqual(costmap_cost_at(costmap, -0.25, -1.75), 10)

    def test_costmap_rejects_unknown_and_outside_points(self):
        costmap = self._costmap(width=2, height=2, resolution=1.0)
        costmap["origin_x"] = 0.0
        costmap["origin_y"] = 0.0
        costmap["data"] = [255, 255, 255, 255]

        self.assertEqual(costmap_cost_at(costmap, 5.0, 5.0), None)
        self.assertIsNone(select_costmap_candidate([(0.5, 0.5)], costmap))

    def test_costmap_rejects_negative_unknown_values(self):
        costmap = self._costmap(width=2, height=2, resolution=1.0)
        costmap["origin_x"] = 0.0
        costmap["origin_y"] = 0.0
        costmap["data"] = [-1, 0, 0, 0]

        self.assertIsNone(select_costmap_candidate([(0.5, 0.5)], costmap))

    def test_costmap_rejects_a_free_but_disconnected_candidate(self):
        costmap = self._costmap(width=5, height=5, resolution=1.0)
        costmap["origin_x"] = 0.0
        costmap["origin_y"] = 0.0
        costmap["data"] = [0] * 25
        for cell_y in range(5):
            costmap["data"][cell_y * 5 + 2] = 254

        selected = select_costmap_candidate(
            [(3.5, 2.5)],
            costmap,
            start=(0.5, 2.5),
        )

        self.assertIsNone(selected)

    def test_costmap_accepts_a_candidate_connected_to_the_robot(self):
        costmap = self._costmap(width=5, height=5, resolution=1.0)
        costmap["origin_x"] = 0.0
        costmap["origin_y"] = 0.0
        costmap["data"] = [0] * 25

        selected = select_costmap_candidate(
            [(3.5, 2.5)],
            costmap,
            start=(0.5, 2.5),
        )

        self.assertEqual(selected, (3.5, 2.5, 0))

    def test_standoff_goal_stops_before_detection_center(self):
        x, y, target_distance = calculate_standoff_goal(
            {"x": 0.0, "y": 0.0},
            target_x=0.0,
            target_y=2.0,
            standoff_distance=0.8,
        )

        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.2)
        self.assertAlmostEqual(target_distance, 2.0)

    def test_standoff_goal_uses_close_default_distance(self):
        _, y, _ = calculate_standoff_goal(
            {"x": 0.0, "y": 0.0},
            target_x=0.0,
            target_y=2.0,
        )

        self.assertAlmostEqual(DEFAULT_STANDOFF_DISTANCE, 0.6)
        self.assertAlmostEqual(y, 1.4)

    def test_standoff_goal_does_not_move_when_already_close(self):
        x, y, target_distance = calculate_standoff_goal(
            {"x": 1.0, "y": 2.0},
            target_x=1.3,
            target_y=2.4,
            standoff_distance=0.8,
        )

        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(target_distance, 0.5)

    def test_standoff_distance_must_be_valid(self):
        with self.assertRaises(ValueError):
            calculate_standoff_goal(
                {"x": 0.0, "y": 0.0},
                target_x=1.0,
                target_y=1.0,
                standoff_distance=-0.1,
            )

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
