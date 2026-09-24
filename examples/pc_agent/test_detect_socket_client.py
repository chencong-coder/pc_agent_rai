import time
import unittest

from .detect_socket_client import DetectBBox3DSocketClient


class DetectionSocketClientTests(unittest.TestCase):
    def test_receive_sequence_distinguishes_frames_without_ros_timestamps(self):
        client = DetectBBox3DSocketClient()
        client.latest_msg = {"stamp": {"sec": 0, "nanosec": 0}}
        client.latest_time = time.time()
        client._receive_sequence = 7

        latest = client.get_latest_with_sequence(max_age=1.0)

        self.assertEqual(
            latest,
            (client.latest_msg, 7, client.latest_time),
        )


if __name__ == "__main__":
    unittest.main()
