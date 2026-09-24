import unittest

from .detection_snapshot import DetectionSnapshotStore


class DetectionSnapshotStoreTests(unittest.TestCase):
    def test_new_round_immediately_invalidates_previous_snapshot(self):
        store = DetectionSnapshotStore()
        first_round = store.begin()
        store.confirm(first_round, [{"class_name": "chair", "x": 1.0}])

        second_round = store.begin()
        state = store.read()

        self.assertEqual(state["round_id"], second_round)
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["detections"], [])

    def test_failed_latest_round_never_falls_back_to_older_success(self):
        store = DetectionSnapshotStore()
        first_round = store.begin()
        store.confirm(first_round, [{"class_name": "chair", "x": 1.0}])
        second_round = store.begin()

        store.fail(second_round, "本轮没有检测到椅子")
        state = store.read()

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["detections"], [])
        self.assertEqual(state["message"], "本轮没有检测到椅子")

    def test_stale_round_cannot_overwrite_the_latest_round(self):
        store = DetectionSnapshotStore()
        first_round = store.begin()
        second_round = store.begin()
        store.confirm(second_round, [{"class_name": "table", "x": 2.0}])

        updated = store.confirm(
            first_round,
            [{"class_name": "chair", "x": 99.0}],
        )
        state = store.read()

        self.assertFalse(updated)
        self.assertEqual(state["detections"][0]["class_name"], "table")

    def test_reset_restores_never_detected_state(self):
        store = DetectionSnapshotStore()
        round_id = store.begin()
        store.confirm(round_id, [{"class_name": "chair"}])

        store.reset()
        state = store.read()

        self.assertIsNone(state["round_id"])
        self.assertEqual(state["status"], "never")
        self.assertEqual(state["detections"], [])


if __name__ == "__main__":
    unittest.main()
