import math
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

from . import localization
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
    def setUp(self):
        self.connectors = []

    def tearDown(self):
        # Every active path must be read-only: no service, cmd_vel or initialpose.
        for connector in self.connectors:
            self.assertEqual(connector.service_calls, [])
            self.assertEqual(connector.messages, [])

    def make_manager(self, clock=None, **kwargs):
        clock = clock or FakeClock()
        connector = FakeConnector()
        self.connectors.append(connector)
        manager = LocalizationManager(
            connector,
            clock=clock,
            wall_clock=clock,
            sleep=clock.sleep,
            **kwargs,
        )
        legacy = patch.object(
            manager,
            "_legacy_ensure_localized",
            side_effect=AssertionError("The retained global search must not run"),
        )
        legacy.start()
        self.addCleanup(legacy.stop)
        return manager, connector, clock

    def localize(self, manager, connector, clock, **seed):
        connector.initial_pose_callback(_pose_message(_covariance(), **seed))
        return manager.ensure_localized()

    def test_covariance_thresholds_and_invalid_values(self):
        self.assertTrue(covariance_is_converged(_covariance()))
        for axis in ("x", "y", "yaw"):
            for value in (0.6, -0.1, math.nan, math.inf):
                with self.subTest(axis=axis, value=value):
                    self.assertFalse(covariance_is_converged(_covariance(**{axis: value})))
        self.assertFalse(covariance_is_converged([0.0] * 10))

    def test_initial_pose_subscription_matches_publisher_qos(self):
        _, connector, _ = self.make_manager()
        self.assertTrue(connector.initial_pose_options["auto_qos_matching"])

    def test_amcl_samples_do_not_localize_before_button_is_clicked(self):
        manager, connector, _ = self.make_manager()
        for seeded in (False, True):
            if seeded:
                connector.initial_pose_callback(_pose_message(_covariance()))
            for _ in range(4):
                connector.callback(_pose_message(_covariance()))
            status = manager.get_status()
            self.assertEqual(status["status"], "waiting")
            self.assertEqual(status["stable_samples"], 0)
            self.assertIsNone(status["pose"])

    def test_initial_pose_immediately_confirms_without_amcl_or_sleep(self):
        manager, connector, clock = self.make_manager(timeout_sec=0.5)
        clock.on_sleep = lambda: self.fail("An available seed must not wait")
        status = self.localize(manager, connector, clock, x=-4.9, y=3.7, yaw=-0.087)
        self.assertEqual(status["status"], "localized")
        self.assertEqual(status["stable_samples"], 0)
        for key, value in (("x", -4.9), ("y", 3.7), ("yaw", -0.087), ("initial_pose_yaw", -0.087)):
            self.assertAlmostEqual(status["pose"][key], value)
        self.assertEqual(status["pose"]["source"], "/initialpose")
        self.assertFalse(status["fresh"])
        self.assertIsNone(status["last_pose_age"])
        self.assertTrue(manager.is_localized())
        self.assertEqual(manager.require_localized()["pose"], status["pose"])

    def test_latest_seed_is_used_instead_of_pre_click_amcl(self):
        manager, connector, _ = self.make_manager()
        connector.initial_pose_callback(_pose_message(_covariance(), x=1.0))
        connector.initial_pose_callback(_pose_message(_covariance(), x=8.0, yaw=-0.1))
        connector.callback(_pose_message(_covariance(), x=99.0))
        pose = manager.ensure_localized()["pose"]
        self.assertEqual(pose["x"], 8.0)
        self.assertAlmostEqual(pose["yaw"], -0.1)
        self.assertEqual(pose["source"], "/initialpose")

    def test_seed_covariance_does_not_gate_initial_coordinates(self):
        manager, connector, _ = self.make_manager()
        connector.initial_pose_callback(_pose_message(_covariance(x=math.nan), x=9.0))
        pose = manager.ensure_localized()["pose"]
        self.assertEqual(pose["x"], 9.0)
        self.assertEqual(pose["source"], "/initialpose")

    def test_live_pose_updates_on_every_amcl_message_and_preserves_seed_heading(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        for x, y, yaw in ((7.0, 8.0, -0.4), (7.2, 8.5, 0.1)):
            clock.now += 0.2
            connector.callback(_pose_message(_covariance(), x=x, y=y, yaw=yaw))
            status = manager.require_localized()
            pose = status["pose"]
            self.assertEqual(status["stable_samples"], 0)
            for key, value in (("x", x), ("y", y), ("yaw", yaw), ("initial_pose_yaw", 0.25)):
                self.assertAlmostEqual(pose[key], value)
            self.assertEqual(pose["source"], "/amcl_pose")
            self.assertEqual(pose["updated_at"], clock.now)

    def test_high_or_invalid_covariance_never_blocks_live_coordinates(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        covariances = [_covariance(x=100.0, y=100.0, yaw=100.0),
                       _covariance(x=math.nan), _covariance(y=math.inf),
                       _covariance(yaw=-1.0), [], None]
        for index, covariance in enumerate(covariances):
            with self.subTest(covariance=covariance):
                clock.now += 0.2
                connector.callback(_pose_message(covariance, x=10.0 + index))
                status = manager.require_localized()
                self.assertEqual(status["pose"]["x"], 10.0 + index)
                self.assertEqual(status["pose"]["source"], "/amcl_pose")
                self.assertEqual(status["pose"]["updated_at"], clock.now)
                self.assertEqual(status["stable_samples"], 0)

    def test_invalid_coordinates_preserve_last_pose_and_update_time(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(), x=8.0))
        original = manager.get_status()
        for axis in ("x", "y"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(axis=axis, value=value):
                    clock.now += 1.0
                    connector.callback(_pose_message(_covariance(), **{axis: value}))
                    status = manager.require_localized()
                    self.assertEqual(status["pose"], original["pose"])
                    self.assertEqual(status["updated_at"], original["updated_at"])
        self.assertFalse(manager.get_status()["fresh"])

    def test_invalid_quaternions_preserve_last_pose_and_update_time(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(), x=8.0))
        original = manager.get_status()
        for rotation in (SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
                         SimpleNamespace(x=math.nan, y=0.0, z=0.0, w=1.0),
                         SimpleNamespace(x=0.0, y=0.0, z=math.inf, w=1.0)):
            with self.subTest(rotation=rotation):
                clock.now += 2.0
                message = _pose_message(_covariance(), x=99.0)
                message.pose.pose.orientation = rotation
                connector.callback(message)
                status = manager.require_localized()
                self.assertEqual(status["pose"], original["pose"])
                self.assertEqual(status["updated_at"], original["updated_at"])
        self.assertFalse(manager.get_status()["fresh"])

    def test_invalid_initial_pose_does_not_replace_confirmed_position(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        original_pose = manager.get_status()["pose"]
        connector.initial_pose_callback(_pose_message(_covariance(), x=math.nan))
        self.assertEqual(manager.require_localized()["pose"], original_pose)

    def test_last_confirmed_pose_persists_while_amcl_is_quiet(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(), x=8.0))
        original_pose = manager.get_status()["pose"]
        clock.now += 3.1
        status = manager.require_localized()
        self.assertFalse(status["fresh"])
        self.assertEqual(status["pose"], original_pose)
        connector.callback(_pose_message(_covariance(x=99.0), x=1.5, y=2.5, yaw=0.5))
        status = manager.require_localized()
        self.assertTrue(status["fresh"])
        self.assertAlmostEqual(status["pose"]["x"], 1.5)
        self.assertEqual(status["stable_samples"], 0)

    def test_navigation_check_does_not_start_any_localization(self):
        manager, _, _ = self.make_manager()
        with patch.object(manager, "start_global_localization") as start:
            with self.assertRaisesRegex(LocalizationError, "获取初始坐标"):
                manager.require_localized()
            start.assert_not_called()

    def test_missing_seed_never_falls_back_to_global_search(self):
        manager, connector, clock = self.make_manager(timeout_sec=3.0)
        clock.on_sleep = lambda: connector.callback(_pose_message(_covariance()))
        with self.assertRaisesRegex(LocalizationError, "未收到 2D Pose Estimate"):
            manager.ensure_localized()
        self.assertEqual(manager.get_status()["status"], "failed")
        self.assertIsNone(manager.get_status()["pose"])
        self.assertLessEqual(clock.now, 2.0)

    def test_seed_arriving_during_wait_immediately_confirms(self):
        manager, connector, clock = self.make_manager(timeout_sec=3.0)
        callbacks = iter([
            lambda: connector.callback(_pose_message(_covariance(), x=99.0)),
            lambda: connector.initial_pose_callback(_pose_message(_covariance(), x=2.1)),
        ])

        def deliver():
            callback = next(callbacks, None)
            self.assertIsNotNone(callback, "A received seed must not wait for AMCL")
            callback()

        clock.on_sleep = deliver
        status = manager.ensure_localized()
        self.assertEqual(status["status"], "localized")
        self.assertEqual(status["stable_samples"], 0)
        self.assertEqual(status["pose"]["x"], 2.1)
        self.assertEqual(status["pose"]["source"], "/initialpose")

    def test_multiple_seeds_arriving_during_wait_confirm_latest_without_amcl(self):
        manager, connector, clock = self.make_manager()
        delivered = False

        def deliver():
            nonlocal delivered
            self.assertFalse(delivered, "The newest available seed must not wait")
            delivered = True
            connector.initial_pose_callback(_pose_message(_covariance(), x=1.0))
            connector.initial_pose_callback(_pose_message(_covariance(), x=2.0))

        clock.on_sleep = deliver
        status = manager.ensure_localized()
        self.assertEqual(status["pose"]["x"], 2.0)
        self.assertEqual(status["pose"]["source"], "/initialpose")
        self.assertEqual(status["stable_samples"], 0)

    def test_manual_acquisition_never_calls_legacy_convergence_helper(self):
        manager, connector, clock = self.make_manager()
        with patch.object(localization, "covariance_is_converged",
                          side_effect=AssertionError("Manual tracking must not check convergence")) as converged:
            self.localize(manager, connector, clock)
            connector.callback(_pose_message(_covariance(x=100.0), x=7.0))
            self.assertEqual(manager.require_localized()["pose"]["x"], 7.0)
            converged.assert_not_called()

    def test_new_seed_requires_click_and_then_immediately_replaces_position(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(), x=5.0))
        connector.initial_pose_callback(_pose_message(_covariance(), x=8.0, yaw=-0.1))
        status = manager.get_status()
        self.assertEqual(status["status"], "waiting")
        self.assertEqual(status["stable_samples"], 0)
        self.assertIsNone(status["pose"])
        self.assertIn("获取初始坐标", status["message"])
        connector.callback(_pose_message(_covariance(), x=99.0))
        self.assertIsNone(manager.get_status()["pose"])
        pose = manager.ensure_localized()["pose"]
        self.assertEqual(pose["x"], 8.0)
        self.assertAlmostEqual(pose["initial_pose_yaw"], -0.1)
        self.assertEqual(pose["source"], "/initialpose")

    def test_repeated_button_and_force_do_not_reset_confirmed_live_pose(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(), x=4.0))
        original_pose = manager.get_status()["pose"]
        self.assertFalse(manager.start_global_localization())
        self.assertEqual(manager.ensure_localized()["pose"], original_pose)
        self.assertEqual(manager.ensure_localized(force=True)["pose"], original_pose)
        self.assertEqual(manager.get_status()["stable_samples"], 0)

    def test_retry_after_missing_seed_uses_new_seed_without_waiting(self):
        manager, connector, clock = self.make_manager(timeout_sec=0.2)
        with self.assertRaises(LocalizationError):
            manager.ensure_localized()
        connector.initial_pose_callback(_pose_message(_covariance(), x=4.0, yaw=-0.1))
        clock.on_sleep = lambda: self.fail("A retry with a seed must not wait")
        status = manager.ensure_localized()
        self.assertEqual(status["status"], "localized")
        self.assertEqual(status["pose"]["x"], 4.0)
        self.assertAlmostEqual(status["pose"]["initial_pose_yaw"], -0.1)

    def test_cancel_seed_wait_does_not_publish_stop_or_accept_late_amcl(self):
        manager, connector, clock = self.make_manager()
        clock.on_sleep = manager.cancel_global_localization
        with self.assertRaises(LocalizationCanceled):
            manager.ensure_localized()
        connector.callback(_pose_message(_covariance()))
        status = manager.get_status()
        self.assertEqual(status["status"], "waiting")
        self.assertIsNone(status["pose"])
        self.assertFalse(manager.cancel_global_localization())

    def test_cancel_after_confirmation_keeps_live_pose(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        original_pose = manager.get_status()["pose"]
        self.assertFalse(manager.cancel_global_localization())
        self.assertEqual(manager.require_localized()["pose"], original_pose)

    def test_background_entry_with_seed_confirms_synchronously_without_thread(self):
        manager, connector, _ = self.make_manager()
        connector.initial_pose_callback(_pose_message(_covariance(), x=3.0))
        with patch.object(localization, "Thread") as thread:
            self.assertTrue(manager.start_global_localization())
            thread.assert_not_called()
        self.assertEqual(manager.require_localized()["pose"]["x"], 3.0)
        self.assertEqual(manager.get_status()["pose"]["source"], "/initialpose")

    def test_background_wait_accepts_late_seed_and_ignores_duplicate_click(self):
        manager, connector, clock = self.make_manager()
        entered, release = Event(), Event()

        def hold():
            entered.set()
            if not release.wait(timeout=5.0):
                raise RuntimeError("Test did not release the background wait")

        clock.on_sleep = hold
        self.assertTrue(manager.start_global_localization())
        thread = manager._background_thread
        try:
            self.assertTrue(entered.wait(timeout=5.0))
            self.assertFalse(manager.start_global_localization())
            connector.initial_pose_callback(_pose_message(_covariance(), x=6.0))
        finally:
            release.set()
            thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(manager.require_localized()["pose"]["x"], 6.0)
        self.assertEqual(manager.get_status()["stable_samples"], 0)

    def test_background_timeout_allows_retry_without_any_global_search(self):
        manager, _, clock = self.make_manager(timeout_sec=0.2)
        for _ in range(2):
            entered, release = Event(), Event()

            def hold():
                entered.set()
                if not release.wait(timeout=5.0):
                    raise RuntimeError("Test did not release the background wait")

            clock.on_sleep = hold
            with self.assertLogs("examples.pc_agent.localization", level="ERROR"):
                self.assertTrue(manager.start_global_localization())
                thread = manager._background_thread
                try:
                    self.assertTrue(entered.wait(timeout=5.0))
                finally:
                    release.set()
                    thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(manager.get_status()["status"], "failed")

    def test_module_entry_points_use_passive_acquisition(self):
        manager, connector, _ = self.make_manager()
        with patch.object(localization, "_active_manager", manager):
            connector.initial_pose_callback(_pose_message(_covariance(), x=7.0))
            self.assertTrue(localization.start_global_localization())
            self.assertEqual(localization.get_localization_status()["pose"]["x"], 7.0)
            self.assertFalse(localization.start_global_localization())
            self.assertFalse(localization.cancel_global_localization())


if __name__ == "__main__":
    unittest.main()
