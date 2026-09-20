import math
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


def _pose_message(covariance, x=1.0, y=2.0, yaw=0.25):
    half_yaw = yaw / 2.0
    return SimpleNamespace(
        pose=SimpleNamespace(
            covariance=covariance,
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=0.0),
                orientation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(half_yaw),
                    w=math.cos(half_yaw),
                ),
            ),
        ),
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
        self.initial_pose_callback = None
        self.initial_pose_options = None
        self.service_calls = []
        self.messages = []

    def register_callback(self, source, callback, **kwargs):
        if source == "/initialpose":
            self.initial_pose_callback = callback
            self.initial_pose_options = kwargs
        else:
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

    def run_localization(
        self,
        manager,
        connector,
        clock,
        messages,
        use_initial_pose=True,
    ):
        if use_initial_pose:
            connector.initial_pose_callback(_pose_message(_covariance()))
        pending = iter(messages)

        def publish_next():
            if manager._localization_mode == "pending":
                return
            message = next(pending, None)
            if message is not None:
                connector.callback(message)

        clock.on_sleep = publish_next
        with patch(
            "examples.pc_agent.localization._make_ros2_message",
            side_effect=lambda payload: SimpleNamespace(payload=payload),
        ):
            return manager.ensure_localized(force=True)

    def test_covariance_thresholds(self):
        self.assertTrue(covariance_is_converged(_covariance()))
        self.assertFalse(covariance_is_converged(_covariance(x=0.6)))
        self.assertFalse(covariance_is_converged(_covariance(y=0.6)))
        self.assertFalse(covariance_is_converged(_covariance(yaw=0.6)))
        self.assertFalse(covariance_is_converged([0.0] * 10))

    def test_converged_samples_do_not_localize_before_button_is_clicked(self):
        manager, connector, clock = self.make_manager()

        for _ in range(4):
            connector.callback(_pose_message(_covariance()))

        status = manager.get_status()
        self.assertEqual(status["status"], "waiting")
        self.assertEqual(status["stable_samples"], 0)
        self.assertIsNone(status["pose"])

    def test_requires_three_consecutive_samples_and_confirms_latest_pose(self):
        manager, connector, clock = self.make_manager(timeout_sec=3.0)
        messages = [
            _pose_message(_covariance(), x=1.0),
            _pose_message(_covariance(), x=2.0),
            _pose_message(_covariance(x=0.8), x=99.0),
            _pose_message(_covariance(), x=3.0),
            _pose_message(_covariance(), x=4.0),
            _pose_message(_covariance(), x=5.0, y=6.0, yaw=0.7),
        ]

        status = self.run_localization(
            manager,
            connector,
            clock,
            messages,
            use_initial_pose=False,
        )

        self.assertEqual(status["status"], "localized")
        self.assertEqual(status["stable_samples"], 3)
        self.assertAlmostEqual(status["pose"]["x"], 5.0)
        self.assertAlmostEqual(status["pose"]["y"], 6.0)
        self.assertAlmostEqual(status["pose"]["yaw"], 0.7)

    def test_confirmed_pose_updates_then_clears_when_quality_is_lost(self):
        manager, connector, clock = self.make_manager(timeout_sec=2.0)
        self.run_localization(
            manager,
            connector,
            clock,
            [_pose_message(_covariance()) for _ in range(3)],
            use_initial_pose=False,
        )

        connector.callback(
            _pose_message(_covariance(), x=7.0, y=8.0, yaw=-0.4)
        )
        self.assertAlmostEqual(manager.get_status()["pose"]["x"], 7.0)

        connector.callback(_pose_message(_covariance(yaw=0.8)))
        status = manager.get_status()
        self.assertEqual(status["status"], "waiting")
        self.assertIsNone(status["pose"])

    def test_confirmed_pose_persists_while_amcl_is_quiet(self):
        manager, connector, clock = self.make_manager(timeout_sec=2.0)
        self.run_localization(
            manager,
            connector,
            clock,
            [_pose_message(_covariance()) for _ in range(3)],
        )

        clock.now += 3.1
        status = manager.get_status()
        self.assertEqual(status["status"], "localized")
        self.assertFalse(status["fresh"])
        self.assertIsNotNone(status["pose"])

        connector.callback(
            _pose_message(_covariance(), x=1.5, y=2.5, yaw=0.5)
        )
        status = manager.get_status()
        self.assertTrue(status["fresh"])
        self.assertAlmostEqual(status["pose"]["x"], 1.5)
        self.assertAlmostEqual(status["pose"]["y"], 2.5)
        self.assertAlmostEqual(status["pose"]["yaw"], 0.5)

    def test_navigation_check_never_starts_global_localization(self):
        manager, connector, _ = self.make_manager()

        with self.assertRaises(LocalizationError):
            manager.require_localized()

        self.assertEqual(connector.service_calls, [])
        self.assertEqual(connector.messages, [])

    def test_manual_initial_pose_confirms_without_rotation(self):
        clock = FakeClock()
        manager, connector, _ = self.make_manager(clock=clock, timeout_sec=2.0)
        status = self.run_localization(
            manager,
            connector,
            clock,
            [],
        )

        self.assertEqual(status["status"], "localized")
        self.assertAlmostEqual(status["pose"]["x"], 1.0)
        self.assertAlmostEqual(status["pose"]["y"], 2.0)
        self.assertAlmostEqual(status["pose"]["yaw"], 0.25)
        self.assertEqual(clock.now, 0.0)
        self.assertEqual(connector.service_calls, [])
        angular_commands = [
            message.payload["angular"]["z"]
            for message, _ in connector.messages
        ]
        self.assertTrue(angular_commands)
        self.assertTrue(all(command == 0.0 for command in angular_commands))

    def test_manual_initial_pose_skips_global_reset(self):
        manager, connector, clock = self.make_manager(timeout_sec=2.0)
        connector.initial_pose_callback(_pose_message(_covariance()))

        status = self.run_localization(
            manager,
            connector,
            clock,
            [],
            use_initial_pose=False,
        )

        self.assertEqual(status["status"], "localized")
        self.assertEqual(connector.service_calls, [])

    def test_manual_pose_stays_valid_while_amcl_covariance_settles(self):
        manager, connector, clock = self.make_manager(timeout_sec=2.0)
        status = self.run_localization(manager, connector, clock, [])

        connector.callback(
            _pose_message(_covariance(x=2.0, y=2.0), x=1.2, y=2.3, yaw=0.4)
        )

        status = manager.get_status()
        self.assertEqual(status["status"], "localized")
        self.assertAlmostEqual(status["pose"]["x"], 1.2)
        self.assertAlmostEqual(status["pose"]["y"], 2.3)
        self.assertAlmostEqual(status["pose"]["yaw"], 0.4)

    def test_initial_pose_clears_previous_stable_samples(self):
        manager, connector, _ = self.make_manager(timeout_sec=2.0)
        connector.callback(_pose_message(_covariance()))
        connector.callback(_pose_message(_covariance()))
        self.assertEqual(manager.get_status()["stable_samples"], 0)

        connector.initial_pose_callback(_pose_message(_covariance()))

        self.assertEqual(manager.get_status()["status"], "waiting")
        self.assertEqual(manager.get_status()["stable_samples"], 0)
        self.assertIsNone(manager.get_status()["pose"])

    def test_initial_pose_subscription_matches_publisher_qos(self):
        _, connector, _ = self.make_manager()

        self.assertTrue(connector.initial_pose_options["auto_qos_matching"])

    def test_latest_manual_initial_pose_is_reused_on_retry(self):
        manager, connector, clock = self.make_manager(timeout_sec=2.0)
        connector.initial_pose_callback(
            _pose_message(_covariance(), x=-4.9, y=2.8, yaw=-0.1)
        )
        connector.initial_pose_callback(
            _pose_message(_covariance(), x=-4.95, y=2.91, yaw=-0.087)
        )
        clock.now = 31.0

        self.assertAlmostEqual(manager._latest_initial_pose["x"], -4.95)
        self.assertAlmostEqual(manager._latest_initial_pose["y"], 2.91)

        status = self.run_localization(
            manager,
            connector,
            clock,
            [],
            use_initial_pose=False,
        )

        self.assertEqual(status["status"], "localized")
        self.assertEqual(connector.service_calls, [])

    def test_missing_initial_pose_falls_back_to_global_rotation(self):
        clock = FakeClock()
        manager, connector, _ = self.make_manager(
            clock=clock,
            timeout_sec=0.5,
        )
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
        self.assertEqual(
            connector.service_calls[0][1]["target"],
            "/reinitialize_global_localization",
        )
        angular_commands = [
            message.payload["angular"]["z"]
            for message, _ in connector.messages
        ]
        self.assertIn(0.2, angular_commands)
        self.assertEqual(angular_commands[-1], 0.0)

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
        self.assertIsNone(manager.get_status()["pose"])
        self.assertEqual(connector.messages[-1][0].payload["angular"]["z"], 0.0)

    def test_background_timeout_allows_a_new_attempt(self):
        manager, _, _ = self.make_manager(timeout_sec=0.2)

        with patch(
            "examples.pc_agent.localization._make_ros2_message",
            side_effect=lambda payload: SimpleNamespace(payload=payload),
        ):
            self.assertTrue(manager.start_global_localization())
            first_thread = manager._background_thread
            first_thread.join(timeout=1.0)
            self.assertFalse(first_thread.is_alive())
            self.assertEqual(manager.get_status()["status"], "failed")

            self.assertTrue(manager.start_global_localization())
            second_thread = manager._background_thread
            second_thread.join(timeout=1.0)
            self.assertFalse(second_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
