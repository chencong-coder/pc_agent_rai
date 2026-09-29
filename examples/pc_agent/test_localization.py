import math
import unittest
from threading import Event
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

    def run_localization(
        self, manager, connector, clock, messages, use_initial_pose=True, force=True
    ):
        if use_initial_pose:
            connector.initial_pose_callback(_pose_message(_covariance()))
        pending = iter(messages)

        def publish_next():
            message = next(pending, None)
            if message is not None:
                connector.callback(message)

        clock.on_sleep = publish_next
        return manager.ensure_localized(force=force)

    def localize(self, manager, connector, clock):
        return self.run_localization(
            manager, connector, clock,
            [_pose_message(_covariance()) for _ in range(3)],
        )

    def test_covariance_thresholds_and_invalid_values(self):
        self.assertTrue(covariance_is_converged(_covariance()))
        for axis in ("x", "y", "yaw"):
            for value in (0.6, -0.1, math.nan, math.inf):
                with self.subTest(axis=axis, value=value):
                    self.assertFalse(
                        covariance_is_converged(_covariance(**{axis: value}))
                    )
        self.assertFalse(covariance_is_converged([0.0] * 10))

    def test_initial_pose_subscription_matches_publisher_qos(self):
        _, connector, _ = self.make_manager()
        self.assertTrue(connector.initial_pose_options["auto_qos_matching"])

    def test_converged_samples_do_not_localize_before_button_is_clicked(self):
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

    def test_initial_pose_alone_never_confirms_localization(self):
        manager, connector, clock = self.make_manager(timeout_sec=0.5)
        connector.initial_pose_callback(_pose_message(_covariance()))
        snapshots = []
        clock.on_sleep = lambda: snapshots.append(manager.get_status())

        with self.assertRaisesRegex(LocalizationError, "未连续稳定"):
            manager.ensure_localized()

        self.assertTrue(snapshots)
        for status in snapshots:
            self.assertEqual(status["stable_samples"], 0)
            self.assertIsNone(status["pose"])
            self.assertFalse(status["fresh"])
        self.assertEqual(manager.get_status()["status"], "failed")

    def test_requires_three_consecutive_samples_and_confirms_latest_amcl_pose(self):
        manager, connector, clock = self.make_manager(timeout_sec=3.0)
        status = self.run_localization(
            manager, connector, clock,
            [
                _pose_message(_covariance(), x=1.0),
                _pose_message(_covariance(), x=2.0),
                _pose_message(_covariance(x=0.8), x=99.0),
                _pose_message(_covariance(), x=3.0),
                _pose_message(_covariance(), x=4.0),
                _pose_message(_covariance(), x=5.0, y=6.0, yaw=0.7),
            ],
        )
        self.assertEqual(status["status"], "localized")
        self.assertEqual(status["stable_samples"], 3)
        self.assertAlmostEqual(status["pose"]["x"], 5.0)
        self.assertAlmostEqual(status["pose"]["y"], 6.0)
        self.assertAlmostEqual(status["pose"]["yaw"], 0.7)
        self.assertAlmostEqual(status["pose"]["initial_pose_yaw"], 0.25)
        self.assertAlmostEqual(clock.now, 1.2)

    def test_live_pose_comes_from_amcl_and_preserves_seed_heading(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        for x, y, yaw in ((7.0, 8.0, -0.4), (7.2, 8.5, 0.1)):
            clock.now += 0.2
            connector.callback(_pose_message(_covariance(), x=x, y=y, yaw=yaw))
            pose = manager.require_localized()["pose"]
            self.assertAlmostEqual(pose["x"], x)
            self.assertAlmostEqual(pose["y"], y)
            self.assertAlmostEqual(pose["yaw"], yaw)
            self.assertAlmostEqual(pose["initial_pose_yaw"], 0.25)
            self.assertEqual(pose["updated_at"], clock.now)

    def test_bad_covariance_requires_reconvergence_without_reseeding(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(yaw=0.8)))
        self.assertEqual(manager.get_status()["status"], "localizing")
        self.assertIsNone(manager.get_status()["pose"])
        with self.assertRaises(LocalizationError):
            manager.require_localized()
        for count in range(1, 4):
            connector.callback(_pose_message(_covariance(), x=7.0))
            self.assertEqual(manager.get_status()["stable_samples"], count)
            self.assertEqual(manager.is_localized(), count == 3)
        self.assertAlmostEqual(manager.get_status()["pose"]["initial_pose_yaw"], 0.25)

    def test_last_confirmed_pose_persists_while_amcl_is_quiet(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        original_pose = manager.get_status()["pose"]
        clock.now += 3.1
        status = manager.get_status()
        self.assertEqual(status["status"], "localized")
        self.assertFalse(status["fresh"])
        self.assertEqual(status["pose"], original_pose)
        connector.callback(_pose_message(_covariance(), x=1.5, y=2.5, yaw=0.5))
        status = manager.get_status()
        self.assertTrue(status["fresh"])
        self.assertAlmostEqual(status["pose"]["x"], 1.5)

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

    def test_late_seed_waits_for_subsequent_amcl_samples(self):
        manager, connector, clock = self.make_manager(timeout_sec=3.0)
        callbacks = iter([
            lambda: connector.callback(_pose_message(_covariance(), x=99.0)),
            lambda: connector.initial_pose_callback(_pose_message(_covariance())),
            *[
                lambda: connector.callback(_pose_message(_covariance(), x=2.1))
                for _ in range(3)
            ],
        ])
        snapshots = []

        def deliver():
            callback = next(callbacks, None)
            if callback:
                callback()
                snapshots.append(manager.get_status())

        clock.on_sleep = deliver
        status = manager.ensure_localized()
        self.assertEqual([s["stable_samples"] for s in snapshots], [0, 0, 1, 2, 3])
        self.assertIsNone(snapshots[1]["pose"])
        self.assertAlmostEqual(status["pose"]["x"], 2.1)

    def test_seed_replacement_during_wait_restarts_only_confirmation(self):
        manager, connector, clock = self.make_manager(timeout_sec=3.0)
        connector.initial_pose_callback(_pose_message(_covariance()))
        actions = iter([
            lambda: connector.callback(_pose_message(_covariance())),
            lambda: connector.callback(_pose_message(_covariance())),
            lambda: connector.initial_pose_callback(
                _pose_message(_covariance(), x=-4.9, yaw=-0.087)
            ),
            *[
                lambda: connector.callback(_pose_message(_covariance(), x=-4.8))
                for _ in range(3)
            ],
        ])
        counts = []

        def deliver():
            callback = next(actions, None)
            if callback:
                callback()
                counts.append(manager.get_status()["stable_samples"])

        clock.on_sleep = deliver
        status = manager.ensure_localized()
        self.assertEqual(counts, [1, 2, 0, 1, 2, 3])
        self.assertAlmostEqual(status["pose"]["x"], -4.8)
        self.assertAlmostEqual(status["pose"]["initial_pose_yaw"], -0.087)

    def test_retry_reuses_latest_seed_without_republishing_it(self):
        manager, connector, clock = self.make_manager(timeout_sec=1.0)
        connector.initial_pose_callback(_pose_message(_covariance(), yaw=0.3))
        connector.initial_pose_callback(_pose_message(_covariance(), yaw=-0.1))
        with self.assertRaises(LocalizationError):
            manager.ensure_localized()
        clock.now += 31.0
        status = self.run_localization(
            manager, connector, clock,
            [_pose_message(_covariance(), x=4.0) for _ in range(3)],
            use_initial_pose=False,
        )
        self.assertEqual(status["status"], "localized")
        self.assertAlmostEqual(status["pose"]["x"], 4.0)
        self.assertAlmostEqual(status["pose"]["initial_pose_yaw"], -0.1)

    def test_sample_gap_breaks_the_consecutive_sequence(self):
        manager, connector, clock = self.make_manager(timeout_sec=10.0)
        count = 0
        samples = []

        def deliver():
            nonlocal count
            count += 1
            if count == 3:
                clock.now += 3.1
            connector.callback(_pose_message(_covariance()))
            samples.append(manager.get_status()["stable_samples"])

        connector.initial_pose_callback(_pose_message(_covariance()))
        clock.on_sleep = deliver
        manager.ensure_localized()
        self.assertEqual(samples, [1, 2, 1, 2, 3])

    def test_invalid_amcl_pose_breaks_the_consecutive_sequence(self):
        manager, connector, clock = self.make_manager()
        status = self.run_localization(
            manager, connector, clock,
            [
                _pose_message(_covariance()),
                _pose_message(_covariance(), x=math.nan),
                *[_pose_message(_covariance()) for _ in range(3)],
            ],
        )
        self.assertEqual(status["status"], "localized")
        self.assertAlmostEqual(clock.now, 1.0)

    def test_new_seed_clears_previous_confirmation(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.initial_pose_callback(_pose_message(_covariance(), x=8.0))
        status = manager.get_status()
        self.assertEqual(status["status"], "waiting")
        self.assertEqual(status["stable_samples"], 0)
        self.assertIsNone(status["pose"])

    def test_repeated_button_does_not_reset_confirmed_live_pose(self):
        manager, connector, clock = self.make_manager()
        self.localize(manager, connector, clock)
        connector.callback(_pose_message(_covariance(), x=4.0))
        original_pose = manager.get_status()["pose"]
        self.assertFalse(manager.start_global_localization())
        self.assertEqual(manager.ensure_localized()["pose"], original_pose)
        self.assertEqual(manager.get_status()["stable_samples"], 3)

    def test_cancel_wait_does_not_publish_a_stop_or_accept_late_samples(self):
        manager, connector, clock = self.make_manager()
        connector.initial_pose_callback(_pose_message(_covariance()))
        clock.on_sleep = manager.cancel_global_localization
        with self.assertRaises(LocalizationCanceled):
            manager.ensure_localized()
        for _ in range(3):
            connector.callback(_pose_message(_covariance()))
        status = manager.get_status()
        self.assertEqual(status["status"], "waiting")
        self.assertIsNone(status["pose"])
        self.assertFalse(manager.cancel_global_localization())

    def test_background_duplicate_start_preserves_inflight_samples(self):
        manager, connector, clock = self.make_manager()
        connector.initial_pose_callback(_pose_message(_covariance()))
        entered, release = Event(), Event()

        def hold():
            entered.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("Test did not release the background wait")

        clock.on_sleep = hold
        self.assertTrue(manager.start_global_localization())
        self.assertTrue(entered.wait(timeout=1.0))
        thread = manager._background_thread
        try:
            connector.callback(_pose_message(_covariance()))
            self.assertFalse(manager.start_global_localization())
            self.assertEqual(manager.get_status()["stable_samples"], 1)
            clock.on_sleep = lambda: connector.callback(_pose_message(_covariance()))
        finally:
            release.set()
            thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(manager.get_status()["status"], "localized")

    def test_background_timeout_allows_retry_without_any_global_search(self):
        manager, _, clock = self.make_manager(timeout_sec=0.2)
        for _ in range(2):
            entered, release = Event(), Event()

            def hold():
                entered.set()
                if not release.wait(timeout=2.0):
                    raise RuntimeError("Test did not release the background wait")

            clock.on_sleep = hold
            with self.assertLogs("examples.pc_agent.localization", level="ERROR"):
                self.assertTrue(manager.start_global_localization())
                self.assertTrue(entered.wait(timeout=1.0))
                thread = manager._background_thread
                release.set()
                thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(manager.get_status()["status"], "failed")


if __name__ == "__main__":
    unittest.main()
