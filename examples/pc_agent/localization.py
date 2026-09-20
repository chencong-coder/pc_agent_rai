# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AMCL localization state and automatic global-localization workflow."""

import logging
import math
import time
from threading import Event, Lock, Thread, current_thread
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

_STATUS_LABELS = {
    "waiting": "等待定位",
    "localizing": "正在定位",
    "localized": "已定位",
    "failed": "定位失败",
}
_INITIAL_POSE_TOPIC = "/initialpose"
_INITIAL_POSE_WAIT_SEC = 2.0

_active_manager = None
_active_manager_lock = Lock()


class LocalizationError(RuntimeError):
    """Raised when AMCL cannot establish a reliable map pose."""


class LocalizationCanceled(LocalizationError):
    """Raised when the user stops an active global-localization attempt."""


def covariance_values(covariance: Sequence[float]) -> tuple[float, float, float]:
    """Return AMCL x, y and yaw variances from a 6x6 covariance matrix."""
    if len(covariance) < 36:
        raise ValueError("AMCL covariance must contain 36 values")

    values = (
        float(covariance[0]),
        float(covariance[7]),
        float(covariance[35]),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("AMCL covariance contains invalid variances")
    return values


def covariance_is_converged(
    covariance: Sequence[float],
    xy_variance_threshold: float = 0.5,
    yaw_variance_threshold: float = 0.5,
) -> bool:
    """Return whether an AMCL covariance is below the configured limits."""
    try:
        x_variance, y_variance, yaw_variance = covariance_values(covariance)
    except (TypeError, ValueError):
        return False
    return (
        x_variance <= xy_variance_threshold
        and y_variance <= xy_variance_threshold
        and yaw_variance <= yaw_variance_threshold
    )


def _make_ros2_message(payload: dict):
    # Keep the state helpers importable for unit tests outside a ROS environment.
    from rai.communication.ros2 import ROS2Message

    return ROS2Message(payload=payload)


def _initial_pose_subscription_options() -> dict:
    """Match RViz's QoS so transient-local initial poses can be replayed."""
    return {"auto_qos_matching": True}


def _yaw_from_quaternion(rotation) -> float:
    components = tuple(
        float(value)
        for value in (rotation.x, rotation.y, rotation.z, rotation.w)
    )
    norm = math.sqrt(sum(value * value for value in components))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("AMCL orientation is invalid")
    x, y, z, w = (value / norm for value in components)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def get_localization_status() -> dict:
    """Return the status of the manager shared by navigation and Streamlit."""
    with _active_manager_lock:
        manager = _active_manager
    if manager is None:
        return {
            "status": "waiting",
            "label": _STATUS_LABELS["waiting"],
            "message": "等待 AMCL 定位数据",
            "fresh": False,
            "stable_samples": 0,
            "required_samples": 3,
            "x_variance": None,
            "y_variance": None,
            "yaw_variance": None,
            "last_pose_age": None,
            "pose": None,
            "updated_at": 0.0,
        }
    return manager.get_status()


def start_global_localization() -> bool:
    """Start one background AMCL global-localization attempt."""
    with _active_manager_lock:
        manager = _active_manager
    if manager is None:
        raise LocalizationError("定位管理器尚未初始化")
    return manager.start_global_localization()


def cancel_global_localization() -> bool:
    """Cancel the background AMCL global-localization attempt."""
    with _active_manager_lock:
        manager = _active_manager
    if manager is None:
        raise LocalizationError("定位管理器尚未初始化")
    return manager.cancel_global_localization()


class LocalizationManager:
    """Confirm AMCL quality and trigger global localization when required."""

    def __init__(
        self,
        connector,
        pose_topic: str = "/amcl_pose",
        global_localization_service: str = "/reinitialize_global_localization",
        cmd_vel_topic: str = "/cmd_vel",
        angular_speed: float = 0.2,
        timeout_sec: float = 45.0,
        service_timeout_sec: float = 5.0,
        freshness_sec: float = 3.0,
        required_samples: int = 3,
        xy_variance_threshold: float = 0.5,
        yaw_variance_threshold: float = 0.5,
        command_period_sec: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if required_samples < 1:
            raise ValueError("required_samples must be at least 1")
        if (
            timeout_sec <= 0.0
            or freshness_sec <= 0.0
        ):
            raise ValueError("localization timeouts must be positive")
        if command_period_sec <= 0.0:
            raise ValueError("command_period_sec must be positive")
        if angular_speed <= 0.0 or not math.isfinite(angular_speed):
            raise ValueError("angular_speed must be a positive finite number")

        self.connector = connector
        self.pose_topic = pose_topic
        self.global_localization_service = global_localization_service
        self.cmd_vel_topic = cmd_vel_topic
        self.angular_speed = angular_speed
        self.timeout_sec = timeout_sec
        self.service_timeout_sec = service_timeout_sec
        self.freshness_sec = freshness_sec
        self.required_samples = required_samples
        self.xy_variance_threshold = xy_variance_threshold
        self.yaw_variance_threshold = yaw_variance_threshold
        self.command_period_sec = command_period_sec
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep

        self._state_lock = Lock()
        self._operation_lock = Lock()
        self._background_lock = Lock()
        self._cancel_event = Event()
        self._background_thread: Optional[Thread] = None
        self._latest_initial_pose: Optional[dict] = None
        self._localization_mode: Optional[str] = None
        self._status = "waiting"
        self._message = "等待 AMCL 定位数据"
        self._stable_samples = 0
        self._last_pose_at: Optional[float] = None
        self._confirmed_pose: Optional[dict] = None
        self._variances: tuple[Optional[float], Optional[float], Optional[float]] = (
            None,
            None,
            None,
        )
        self._updated_at = self._wall_clock()

        self._callback_id = self.connector.register_callback(
            self.pose_topic,
            self._on_amcl_pose,
            raw=True,
            msg_type="geometry_msgs/msg/PoseWithCovarianceStamped",
        )
        self._initial_pose_callback_id = self.connector.register_callback(
            _INITIAL_POSE_TOPIC,
            self._on_initial_pose,
            raw=True,
            msg_type="geometry_msgs/msg/PoseWithCovarianceStamped",
            **_initial_pose_subscription_options(),
        )

        global _active_manager
        with _active_manager_lock:
            _active_manager = self

    def _set_status_locked(self, status: str, message: str) -> None:
        self._status = status
        self._message = message
        self._updated_at = self._wall_clock()

    def _clear_confirmation_locked(self) -> None:
        self._stable_samples = 0
        self._confirmed_pose = None

    def _confirm_pose_locked(self, x: float, y: float, yaw: float) -> None:
        self._confirmed_pose = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "updated_at": self._wall_clock(),
        }

    def _confirm_initial_pose_locked(self, initial_pose: dict) -> None:
        self._localization_mode = "manual"
        self._last_pose_at = self._clock()
        self._stable_samples = self.required_samples
        self._confirm_pose_locked(
            initial_pose["x"],
            initial_pose["y"],
            initial_pose["yaw"],
        )
        self._set_status_locked(
            "localized",
            "已使用 RViz 2D Pose Estimate 作为初始位姿",
        )

    def _is_localized_locked(self, _now: float) -> bool:
        return (
            self._status == "localized"
            and self._last_pose_at is not None
            and self._stable_samples >= self.required_samples
            and self._confirmed_pose is not None
        )

    def _on_amcl_pose(self, message) -> None:
        try:
            payload = getattr(message, "payload", message)
            pose_with_covariance = payload.pose
            pose = pose_with_covariance.pose
            position = pose.position
            self.observe_pose(
                pose_with_covariance.covariance,
                float(position.x),
                float(position.y),
                _yaw_from_quaternion(pose.orientation),
            )
        except Exception as exc:
            logger.warning("无法读取 %s 位姿: %s", self.pose_topic, exc)
            self.observe_pose((), math.nan, math.nan, math.nan)

    def _on_initial_pose(self, message) -> None:
        """Store the complete latest RViz pose for the next attempt."""
        try:
            payload = getattr(message, "payload", message)
            pose_with_covariance = payload.pose
            pose = pose_with_covariance.pose
            position = pose.position
            initial_pose = {
                "x": float(position.x),
                "y": float(position.y),
                "yaw": _yaw_from_quaternion(pose.orientation),
                "received_at": self._clock(),
            }
            if not all(
                math.isfinite(initial_pose[key])
                for key in ("x", "y", "yaw", "received_at")
            ):
                raise ValueError("初始位姿包含非有限数值")
        except Exception as exc:
            logger.warning("无法读取 %s 初始位姿: %s", _INITIAL_POSE_TOPIC, exc)
            return

        with self._state_lock:
            self._latest_initial_pose = initial_pose
            self._clear_confirmation_locked()
            logger.info(
                "收到最新 %s：x=%.3f, y=%.3f, yaw=%.3f",
                _INITIAL_POSE_TOPIC,
                initial_pose["x"],
                initial_pose["y"],
                initial_pose["yaw"],
            )
            if self._status == "localized":
                self._set_status_locked(
                    "waiting",
                    "收到新的 RViz 初始位姿，请点击“自动定位”确认",
                )
            elif self._status == "localizing":
                self._confirm_initial_pose_locked(initial_pose)

    def observe_pose(
        self,
        covariance: Sequence[float],
        x: float,
        y: float,
        yaw: float,
        received_at: Optional[float] = None,
    ) -> None:
        """Record one complete AMCL pose and covariance sample."""
        now = self._clock() if received_at is None else received_at
        pose_is_valid = all(math.isfinite(value) for value in (x, y, yaw))
        try:
            variances = covariance_values(covariance)
        except (TypeError, ValueError):
            variances = (None, None, None)
        converged = (
            pose_is_valid
            and covariance_is_converged(
                covariance,
                self.xy_variance_threshold,
                self.yaw_variance_threshold,
            )
        )

        with self._state_lock:
            if self._status == "localizing" and self._localization_mode == "pending":
                # Do not confirm stale AMCL data while deciding whether this
                # attempt uses RViz's seed or global rotating localization.
                return
            if (
                self._last_pose_at is None
                or now - self._last_pose_at > self.freshness_sec
            ):
                self._stable_samples = 0
            self._last_pose_at = now
            self._variances = variances

            if self._status == "localizing" and converged:
                self._stable_samples += 1
                if self._stable_samples >= self.required_samples:
                    self._stable_samples = self.required_samples
                    self._confirm_pose_locked(x, y, yaw)
                    self._set_status_locked(
                        "localized",
                        f"AMCL 已连续 {self.required_samples} 帧收敛",
                    )
                else:
                    mode_label = (
                        "AMCL 初始位姿收敛中"
                        if self._localization_mode == "manual"
                        else "AMCL 全局定位中"
                    )
                    self._set_status_locked(
                        "localizing",
                        f"{mode_label}（稳定样本 "
                        f"{self._stable_samples}/{self.required_samples}）",
                    )
            elif self._status == "localizing":
                self._stable_samples = 0
                self._confirmed_pose = None
                mode_label = (
                    "AMCL 初始位姿收敛中"
                    if self._localization_mode == "manual"
                    else "AMCL 全局定位中"
                )
                self._set_status_locked(
                    "localizing",
                    f"{mode_label}（位姿质量尚未收敛）",
                )
            elif self._status == "localized" and converged:
                self._stable_samples = self.required_samples
                self._confirm_pose_locked(x, y, yaw)
            elif (
                self._status == "localized"
                and self._localization_mode == "manual"
                and pose_is_valid
            ):
                # An explicit RViz pose is already an accepted starting pose.
                # Continue refreshing the live position without requiring the
                # global-localization covariance gate.
                self._stable_samples = self.required_samples
                self._confirm_pose_locked(x, y, yaw)
            elif self._status == "localized":
                self._clear_confirmation_locked()
                self._set_status_locked(
                    "waiting",
                    "AMCL 定位质量已失效，请重新点击“自动定位”",
                )

    def is_localized(self) -> bool:
        now = self._clock()
        with self._state_lock:
            return self._is_localized_locked(now)

    def get_status(self) -> dict:
        now = self._clock()
        with self._state_lock:
            fresh = (
                self._last_pose_at is not None
                and now - self._last_pose_at <= self.freshness_sec
            )
            age = None
            if self._last_pose_at is not None:
                age = max(0.0, now - self._last_pose_at)
            x_variance, y_variance, yaw_variance = self._variances
            return {
                "status": self._status,
                "label": _STATUS_LABELS[self._status],
                "message": self._message,
                "fresh": fresh,
                "stable_samples": self._stable_samples,
                "required_samples": self.required_samples,
                "x_variance": x_variance,
                "y_variance": y_variance,
                "yaw_variance": yaw_variance,
                "last_pose_age": age,
                "pose": dict(self._confirmed_pose)
                if self._is_localized_locked(now)
                else None,
                "updated_at": self._updated_at,
            }

    def _publish_rotation(self, angular_z: float) -> None:
        payload = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": angular_z},
        }
        self.connector.send_message(
            _make_ros2_message(payload),
            target=self.cmd_vel_topic,
            msg_type="geometry_msgs/msg/Twist",
        )

    def _mark_failed(self, message: str) -> None:
        with self._state_lock:
            self._clear_confirmation_locked()
            self._set_status_locked("failed", message)

    def require_localized(self) -> dict:
        """Return current quality state or reject navigation without moving."""
        if self.is_localized():
            return self.get_status()
        status = self.get_status()
        detail = status.get("message") or status.get("label")
        raise LocalizationError(
            f"当前 AMCL 定位不可用（{detail}），请先在页面点击“自动定位”"
        )

    def _background_localization(self) -> None:
        try:
            self.ensure_localized(force=True)
        except LocalizationCanceled:
            logger.info("AMCL 自动定位已取消")
        except LocalizationError:
            logger.exception("AMCL 自动定位失败")
        finally:
            with self._background_lock:
                if self._background_thread is current_thread():
                    self._background_thread = None

    def start_global_localization(self) -> bool:
        """Start global localization without blocking the Streamlit page."""
        with self._background_lock:
            if (
                self._background_thread is not None
                and self._background_thread.is_alive()
            ):
                return False
            self._cancel_event.clear()
            with self._state_lock:
                self._clear_confirmation_locked()
                self._last_pose_at = None
                self._localization_mode = "pending"
                self._set_status_locked(
                    "localizing",
                    "正在读取最新初始位姿，准备 AMCL 定位",
                )
            self._background_thread = Thread(
                target=self._background_localization,
                name="amcl-global-localization",
                daemon=True,
            )
            self._background_thread.start()
            return True

    def cancel_global_localization(self) -> bool:
        """Cancel an active attempt and send an immediate zero velocity."""
        with self._state_lock:
            active = self._status == "localizing"
            if active:
                self._clear_confirmation_locked()
                self._set_status_locked("waiting", "自动定位已取消")
        if not active:
            return False

        self._cancel_event.set()
        try:
            self._publish_rotation(0.0)
        except Exception:
            logger.exception("取消自动定位时发送停止指令失败")
        return True

    def ensure_localized(self, force: bool = False) -> dict:
        """Confirm AMCL from RViz or fall back to global rotating localization."""
        with self._operation_lock:
            if not force and self.is_localized():
                return self.get_status()
            if self._cancel_event.is_set():
                raise LocalizationCanceled("自动定位已取消")

            with self._state_lock:
                self._clear_confirmation_locked()
                self._last_pose_at = None
                self._localization_mode = "pending"
                self._set_status_locked(
                    "localizing",
                    "正在定位，等待 AMCL 收敛",
                )

            error: Optional[LocalizationError] = None
            localized = False
            try:
                with self._state_lock:
                    manual_seed = (
                        dict(self._latest_initial_pose)
                        if self._latest_initial_pose is not None
                        else None
                    )

                # The ROS callback is asynchronous. If the user published
                # 2D Pose Estimate just before clicking, allow that message
                # to arrive before falling back to global localization.
                if manual_seed is None:
                    seed_wait_deadline = self._clock() + _INITIAL_POSE_WAIT_SEC
                    while self._clock() < seed_wait_deadline:
                        if self._cancel_event.is_set():
                            error = LocalizationCanceled("自动定位已取消")
                            break
                        with self._state_lock:
                            if self._latest_initial_pose is not None:
                                manual_seed = dict(self._latest_initial_pose)
                                break
                        remaining = seed_wait_deadline - self._clock()
                        self._sleep(min(self.command_period_sec, remaining))

                if error is not None:
                    pass
                elif manual_seed is not None:
                    with self._state_lock:
                        self._confirm_initial_pose_locked(manual_seed)
                    logger.info(
                        "读取最新 %s：x=%.3f, y=%.3f, yaw=%.3f，直接作为定位起点",
                        _INITIAL_POSE_TOPIC,
                        manual_seed["x"],
                        manual_seed["y"],
                        manual_seed["yaw"],
                    )
                else:
                    with self._state_lock:
                        self._localization_mode = "global"
                        self._last_pose_at = None
                    logger.info(
                        "未收到最新 %s，调用 AMCL 全局定位并开始原地旋转",
                        _INITIAL_POSE_TOPIC,
                    )
                    self.connector.service_call(
                        _make_ros2_message({}),
                        target=self.global_localization_service,
                        msg_type="std_srvs/srv/Empty",
                        timeout_sec=self.service_timeout_sec,
                    )

                if error is None:
                    deadline = self._clock() + self.timeout_sec
                    while self._clock() < deadline:
                        if self._cancel_event.is_set():
                            error = LocalizationCanceled("自动定位已取消")
                            break
                        if self.is_localized():
                            localized = True
                            break
                        if manual_seed is None:
                            self._publish_rotation(self.angular_speed)
                        remaining = deadline - self._clock()
                        if remaining > 0.0:
                            self._sleep(min(self.command_period_sec, remaining))

                if error is None and not localized:
                    localized = self.is_localized()
                if error is None and not localized:
                    error = LocalizationError(
                        f"AMCL 在 {self.timeout_sec:g} 秒内未收敛"
                    )
            except Exception as exc:
                if isinstance(exc, LocalizationError):
                    error = exc
                else:
                    error = LocalizationError(
                        "调用 AMCL 全局定位或旋转小车失败"
                        f"（{self.global_localization_service}）：{exc}"
                    )
            finally:
                try:
                    self._publish_rotation(0.0)
                except Exception as exc:
                    logger.exception("停止 AMCL 定位旋转失败")
                    if error is None:
                        error = LocalizationError(f"定位结束后无法停止小车：{exc}")

            if error is None and not self.is_localized():
                error = LocalizationError("AMCL 定位结果在停车后失效")
            if isinstance(error, LocalizationCanceled):
                with self._state_lock:
                    self._clear_confirmation_locked()
                    self._localization_mode = None
                    self._set_status_locked("waiting", str(error))
                raise error
            if error is not None:
                self._mark_failed(str(error))
                with self._state_lock:
                    self._localization_mode = None
                raise error

            return self.get_status()
