import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .localization import (
    LocalizationCanceled,
    LocalizationError,
    LocalizationManager,
    covariance_is_converged,
)


def _covariance(x=0.1, y=0.1, yaw=0.1):
    values = [0.0] * 36
    values[0] = x
    values[7] = y
    values[35] = yaw
    return values


def _pose_message(covariance):
    return SimpleNamespace(
        pose=SimpleNamespace(covariance=covariance),
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration
        if self.on_sleep is not None:
            self.on_sleep()


class FakeConnector:
    def __init__(self):
        self.callback = None
        self.service_calls = []
        self.messages = []

    def register_callback(self, source, callback, **kwargs):
        self.callback = callback
        return "callback-1"

    def service_call(self, message, **kwargs):
        self.service_calls.append((message, kwargs))
        return SimpleNamespace(payload={})

    def send_message(self, message, **kwargs):
        self.messages.append((message, kwargs))


class LocalizationQualityTests(unittest.TestCase):
    def make_manager(self, clock=None, **kwargs):
        clock = clock or FakeClock()
        connector = FakeConnector()
        manager = LocalizationManager(
            connector,
            clock=clock,
            wall_clock=clock,
            sleep=clock.sleep,
            **kwargs,
        )
        return manager, connector, clock

    def test_covariance_thresholds(self):
        self.assertTrue(covariance_is_converged(_covariance()))
        self.assertFalse(covariance_is_converged(_covariance(x=0.6)))
        self.assertFalse(covariance_is_converged(_covariance(y=0.6)))
        self.assertFalse(covariance_is_converged(_covariance(yaw=0.6)))
        self.assertFalse(covariance_is_converged([0.0] * 10))

    def test_requires_three_consecutive_fresh_samples(self):
        manager, connector, clock = self.make_manager()

        connector.callback(_pose_message(_covariance()))
        connector.callback(_pose_message(_covariance()))
        self.assertEqual(manager.get_status()["status"], "waiting")

        connector.callback(_pose_message(_covariance(x=0.8)))
        connector.callback(_pose_message(_covariance()))
        connector.callback(_pose_message(_covariance()))
        self.assertEqual(manager.get_status()["status"], "waiting")

        connector.callback(_pose_message(_covariance()))
        self.assertEqual(manager.get_status()["status"], "localized")

        clock.now += 3.1
        self.assertEqual(manager.get_status()["status"], "waiting")

    def test_navigation_check_never_starts_global_localization(self):
        manager, connector, _ = self.make_manager()

        with self.assertRaises(LocalizationError):
            manager.require_localized()

        self.assertEqual(connector.service_calls, [])
        self.assertEqual(connector.messages, [])

    def test_global_localization_rotates_until_converged_then_stops(self):
        clock = FakeClock()
        manager, connector, _ = self.make_manager(clock=clock, timeout_sec=2.0)
        clock.on_sleep = lambda: connector.callback(
            _pose_message(_covariance())
        )

        with patch(
            "examples.pc_agent.localization._make_ros2_message",
            side_effect=lambda payload: SimpleNamespace(payload=payload),
        ):
            status = manager.ensure_localized()

        self.assertEqual(status["status"], "localized")
        self.assertEqual(len(connector.service_calls), 1)
        angular_commands = [
            message.payload["angular"]["z"]
            for message, _ in connector.messages
        ]
        self.assertIn(0.2, angular_commands)
        self.assertEqual(angular_commands[-1], 0.0)

    def test_global_localization_timeout_still_stops(self):
        manager, connector, _ = self.make_manager(timeout_sec=0.5)

        with patch(
            "examples.pc_agent.localization._make_ros2_message",
            side_effect=lambda payload: SimpleNamespace(payload=payload),
        ):
            with self.assertRaises(LocalizationError):
                manager.ensure_localized()

        self.assertEqual(manager.get_status()["status"], "failed")
        self.assertEqual(connector.messages[-1][0].payload["angular"]["z"], 0.0)

    def test_cancel_global_localization_stops_and_returns_to_waiting(self):
        clock = FakeClock()
        manager, connector, _ = self.make_manager(clock=clock, timeout_sec=2.0)
        clock.on_sleep = manager.cancel_global_localization

        with patch(
            "examples.pc_agent.localization._make_ros2_message",
            side_effect=lambda payload: SimpleNamespace(payload=payload),
        ):
            with self.assertRaises(LocalizationCanceled):
                manager.ensure_localized(force=True)

        self.assertEqual(manager.get_status()["status"], "waiting")
        self.assertEqual(connector.messages[-1][0].payload["angular"]["z"], 0.0)


if __name__ == "__main__":
    unittest.main()
