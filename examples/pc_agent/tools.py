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

"""
PC Agent Tools - 用于 PC 端与大模型交互的 ROS 2 工具

三个核心工具:
1. GetDetectionsTool        - 读 Orin 的 VoteNet 3D 检测结果
2. NavigateToCoordinatesTool - 向 Orin 的 Nav2 发送导航目标
3. CancelNavigationTool     - 取消当前导航任务
"""

import json
import logging
import math
import time
from threading import Lock
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from rai.communication.ros2 import ROS2Message
from rai.communication.ros2.connectors import ROS2Connector

from .detect_socket_client import DetectBBox3DSocketClient
from .detection_selection import (
    CLASS_NAMES_ZH,
    direction_from_robot_frame,
    select_one_to_one_matches,
    select_detection_targets,
    summarize_detection_directions,
)
from .detection_snapshot import DetectionSnapshotStore
from .localization import LocalizationError
from .navigation_heading import (
    calculate_detection_standoff,
    generate_standoff_candidates,
    normalize_costmap,
    resolve_navigation_yaw,
    select_costmap_candidate,
)

logger = logging.getLogger(__name__)

# 模块级检测缓存 — 持续订阅，每次 _run 只读缓存
_detection_cache: dict = {}
_detection_lock = Lock()
_socket_clients: dict[tuple[str, int], DetectBBox3DSocketClient] = {}
_socket_clients_lock = Lock()
_detection_snapshots = DetectionSnapshotStore()
_active_navigation_action_id: Optional[str] = None
_navigation_action_lock = Lock()
_navigation_status: dict = {
    "status": "idle",
    "action_id": None,
    "event_id": None,
    "x": None,
    "y": None,
    "message": "",
    "result_code": None,
    "updated_at": 0.0,
}

def get_navigation_status() -> dict:
    """Return a thread-safe snapshot of the latest Nav2 navigation status."""
    with _navigation_action_lock:
        return dict(_navigation_status)


def _mark_navigation_canceling(action_id: str) -> None:
    with _navigation_action_lock:
        if _active_navigation_action_id != action_id:
            return
        _navigation_status.update(
            status="canceling",
            message="正在取消导航",
            updated_at=time.time(),
        )


def _finish_navigation(
    action_id: str,
    status: str,
    message: str,
    result_code: Optional[int] = None,
) -> None:
    global _active_navigation_action_id
    with _navigation_action_lock:
        if _active_navigation_action_id != action_id:
            return
        _active_navigation_action_id = None
        _navigation_status.update(
            status=status,
            action_id=action_id,
            event_id=action_id,
            message=message,
            result_code=result_code,
            updated_at=time.time(),
        )


def _mark_navigation_failed(x: float, y: float, message: str) -> None:
    global _active_navigation_action_id
    event_id = f"navigation-{time.time_ns()}"
    with _navigation_action_lock:
        _active_navigation_action_id = None
        _navigation_status.update(
            status="failed",
            action_id=None,
            event_id=event_id,
            x=float(x),
            y=float(y),
            message=message,
            result_code=None,
            updated_at=time.time(),
        )


def _handle_navigation_done(action_id: str, future) -> None:
    """Translate the Nav2 action result into the status consumed by Streamlit."""
    try:
        response = future.result()
        result_code = int(getattr(response, "status", 0))
    except Exception as exc:
        _finish_navigation(
            action_id,
            "failed",
            f"导航结果读取失败：{exc}",
        )
        return

    terminal_states = {
        4: ("completed", "导航完成"),
        5: ("canceled", "导航已取消"),
        6: ("failed", "导航失败，Nav2 已终止任务"),
    }
    status, message = terminal_states.get(
        result_code,
        ("failed", f"导航结束，Nav2 状态码 {result_code}"),
    )
    if status == "failed":
        result = getattr(response, "result", None)
        error_code = getattr(result, "error_code", None)
        error_message = str(getattr(result, "error_msg", "") or "").strip()
        details = []
        if error_code not in (None, 0):
            details.append(f"error_code={error_code}")
        if error_message:
            details.append(error_message)
        if details:
            message = f"{message}（{'；'.join(details)}）"
        else:
            message = (
                f"{message}（Action 状态码 {result_code}；Nav2 未返回详细原因，"
                "请查看 planner_server/controller_server 日志）"
            )
    _finish_navigation(action_id, status, message, result_code)


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def _get_socket_client(host: str, port: int) -> DetectBBox3DSocketClient:
    key = (host, port)
    with _socket_clients_lock:
        client = _socket_clients.get(key)
        if client is None:
            client = DetectBBox3DSocketClient(host=host, port=port)
            client.start()
            _socket_clients[key] = client
        return client


# ─── Data Models ──────────────────────────────────────────────────────────

class DetectionObject(BaseModel):
    class_name: str = Field(description="类别: bed, chair, table...")
    x: float = Field(description="x (m)")
    y: float = Field(description="y (m)")
    z: float = Field(description="z (m)")
    size_x: float = Field(default=0.0, description="检测框 x 尺寸 (m)")
    size_y: float = Field(default=0.0, description="检测框 y 尺寸 (m)")
    confidence: float = Field(default=0.0)
    direction: str = Field(default="方向未知")
    confirmed_hits: int = Field(default=1)


def get_detection_snapshot() -> list[DetectionObject]:
    state = _detection_snapshots.read()
    if state["status"] != "confirmed":
        return []
    return state["detections"]


def get_latest_detection_round() -> dict:
    """Return the latest detection call and its matching structured snapshot."""
    return _detection_snapshots.read()


def clear_detection_history() -> None:
    """Clear both the latest detection conversation state and its snapshot."""
    _detection_snapshots.reset()


class DetectionTransformError(RuntimeError):
    """Raised when a detection cannot be converted into the navigation frame."""


class DetectionTrack:
    """Track one same-class object across several detection frames."""

    def __init__(self, detection: DetectionObject, timestamp: float):
        self.class_name = detection.class_name
        self.samples = [(timestamp, detection)]
        self.last_seen = timestamp
        self.missed_frames = 0

    @property
    def last_detection(self) -> DetectionObject:
        return self.samples[-1][1]

    def add(
        self, detection: DetectionObject, timestamp: float, window_size: int
    ) -> None:
        self.samples.append((timestamp, detection))
        self.samples = self.samples[-window_size:]
        self.last_seen = timestamp
        self.missed_frames = 0

    def smooth(self) -> DetectionObject:
        recent = [detection for _, detection in self.samples[-3:]]
        def median(values):
            return sorted(values)[len(values) // 2]
        return DetectionObject(
            class_name=self.class_name,
            x=median([d.x for d in recent]),
            y=median([d.y for d in recent]),
            z=median([d.z for d in recent]),
            size_x=median([d.size_x for d in recent]),
            size_y=median([d.size_y for d in recent]),
            confidence=sum(d.confidence for d in recent) / len(recent),
            direction=self.last_detection.direction,
            confirmed_hits=len(self.samples),
        )


class DetectionStabilizer:
    """Confirm objects after repeated spatially consistent observations."""

    def __init__(
        self,
        min_hits: int = 3,
        window_size: int = 3,
        window_seconds: float = 2.0,
        match_distance: float = 0.5,
        max_missed_frames: int = 5,
    ):
        self.min_hits = min_hits
        self.window_size = window_size
        self.window_seconds = window_seconds
        self.match_distance = match_distance
        self.max_missed_frames = max_missed_frames
        self.tracks: list[DetectionTrack] = []
        self.last_payload_time = 0.0

    def update(
        self, detections: list[DetectionObject], timestamp: float
    ) -> list[DetectionObject]:
        for track in self.tracks:
            track.missed_frames += 1

        unmatched = set(range(len(detections)))
        candidates = []
        for track_index, track in enumerate(self.tracks):
            previous = track.last_detection
            for detection_index in unmatched:
                detection = detections[detection_index]
                if detection.class_name.lower() != track.class_name.lower():
                    continue
                distance = math.hypot(
                    detection.x - previous.x,
                    detection.y - previous.y,
                )
                if distance <= self.match_distance:
                    candidates.append((distance, track_index, detection_index))

        matches = select_one_to_one_matches(candidates)
        for _, track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.add(
                detections[detection_index], timestamp, self.window_size
            )
            unmatched.remove(detection_index)

        for detection_index in unmatched:
            self.tracks.append(
                DetectionTrack(detections[detection_index], timestamp)
            )

        self.tracks = [
            track for track in self.tracks
            if track.missed_frames <= self.max_missed_frames
            and timestamp - track.last_seen <= self.window_seconds * 2
        ]
        self.last_payload_time = timestamp

        confirmed = []
        for track in self.tracks:
            if len(track.samples) < self.min_hits:
                continue
            recent = track.samples[-self.min_hits:]
            if recent[-1][0] - recent[0][0] > self.window_seconds:
                continue
            if track.missed_frames == 0:
                confirmed.append(track.smooth())
        return confirmed


# ─── Tool: 获取检测结果 ───────────────────────────────────────────────────

class GetDetectionsToolInput(BaseModel):
    object_class: Optional[str] = Field(
        default=None,
        description="按类别过滤，如 'chair'。不填返回全部。"
    )


class GetDetectionsTool(BaseTool):
    """读取 Orin VoteNet 发布的 3D 室内检测结果。

    默认通过 TCP socket 读取 Orin Foxy 侧 bridge 发来的 JSON，
    避免 Humble 容器直接订阅 Foxy ROS 2 话题。
    """

    name: str = "get_detections"
    description: str = (
        "获取小车 VoteNet 检测到的周围物体及其 3D 坐标。"
        "可选参数 object_class 按类别过滤（如 'chair'）。"
        "仅在检测坐标成功转换到 map 后返回物体类别、地图坐标(x,y,z)、置信度。"
    )
    args_schema: Type[GetDetectionsToolInput] = GetDetectionsToolInput

    connector: ROS2Connector = Field(..., exclude=True)
    topic: str = Field(default="/detect_bbox3d")
    detection_source: str = Field(default="socket", description="socket 或 ros")
    socket_host: str = Field(default="127.0.0.1")
    socket_port: int = Field(default=8765)
    target_frame: str = Field(default="map", description="TF 变换目标坐标系")
    cache_max_age: float = Field(default=10.0, description="缓存有效时间(秒)")
    timeout_sec: float = Field(default=15.0)
    confirmation_hits: int = Field(default=3)
    confirmation_window: int = Field(default=3)
    confirmation_window_seconds: float = Field(default=2.0)
    confirmation_distance: float = Field(default=0.5)
    confirmation_max_missed_frames: int = Field(default=5)

    def _ensure_subscribed(self):
        import rclpy
        from vision_msgs.msg import Detection3DArray

        # 独立 rclpy 上下文 — 避免跟 connector 的 executor 冲突
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("rai_detect_sub")
        cache_entry = {"payload": None, "timestamp": 0.0, "count": 0}

        def _on_detection(msg: Detection3DArray):
            cache_entry["payload"] = msg
            cache_entry["timestamp"] = time.time()
            cache_entry["count"] += 1

        node.create_subscription(Detection3DArray, self.topic, _on_detection, 10)
        logger.info(f"已创建订阅 {self.topic}，等待消息...")

        # spin_once 循环 — 跟 ros2 topic echo 一样
        warm_start = time.time()
        while time.time() - warm_start < self.timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.2)
            if cache_entry["payload"] is not None:
                logger.info(f"收到 {cache_entry['count']} 条消息")
                break
        node.destroy_node()

        return cache_entry

    @staticmethod
    def _parse_detection3d_array(payload) -> list[DetectionObject]:
        detections = []
        for det in payload.detections:
            if det.results:
                best = det.results[0]
                score = float(best.hypothesis.score)
                class_id = best.hypothesis.class_id.strip()
                x = float(best.pose.pose.position.x)
                y = float(best.pose.pose.position.y)
                z = float(best.pose.pose.position.z)
                size_x = float(getattr(det.bbox.size, "x", 0.0))
                size_y = float(getattr(det.bbox.size, "y", 0.0))

                # 过滤无效检测: score=0, class 为空, nan/inf 坐标
                if score <= 0.0 or not class_id:
                    continue
                if math.isnan(x) or math.isinf(x):
                    continue
                if math.isnan(y) or math.isinf(y):
                    continue
                if math.isnan(z) or math.isinf(z):
                    continue

                detections.append(DetectionObject(
                    class_name=class_id,
                    x=x, y=y, z=z,
                    size_x=size_x if math.isfinite(size_x) else 0.0,
                    size_y=size_y if math.isfinite(size_y) else 0.0,
                    confidence=score,
                ))
        return detections

    @staticmethod
    def _parse_socket_payload(payload: dict) -> list[DetectionObject]:
        detections = []
        for det in payload.get("detections", []):
            class_id = str(det.get("class_id", "")).strip()
            score = float(det.get("score", 0.0) or 0.0)
            center = det.get("center", {}) or {}
            size = det.get("size", {}) or {}

            x = float(center.get("x", 0.0) or 0.0)
            y = float(center.get("y", 0.0) or 0.0)
            z = float(center.get("z", 0.0) or 0.0)
            size_x = float(size.get("x", 0.0) or 0.0)
            size_y = float(size.get("y", 0.0) or 0.0)

            if score <= 0.0 or not class_id:
                continue
            if math.isnan(x) or math.isinf(x):
                continue
            if math.isnan(y) or math.isinf(y):
                continue
            if math.isnan(z) or math.isinf(z):
                continue

            detections.append(DetectionObject(
                class_name=class_id,
                x=x, y=y, z=z,
                size_x=size_x if math.isfinite(size_x) else 0.0,
                size_y=size_y if math.isfinite(size_y) else 0.0,
                confidence=score,
                direction=str(
                    det.get("relative_direction", det.get("direction", "方向未知"))
                ),
                confirmed_hits=int(det.get("confirmed_hits", 1) or 1),
            ))
        return detections

    @staticmethod
    def _payload_timestamp(payload: dict, fallback: float | None = None) -> float:
        """Use the producer stamp when valid, otherwise local receive time."""
        stamp = payload.get("stamp", {}) or {}
        try:
            candidate = float(stamp.get("sec", 0.0)) + (
                float(stamp.get("nanosec", 0.0)) / 1e9
            )
        except (TypeError, ValueError):
            candidate = 0.0
        if math.isfinite(candidate) and candidate > 0.0:
            return candidate
        local_time = float(fallback if fallback is not None else time.time())
        return local_time if math.isfinite(local_time) else time.time()

    @staticmethod
    def _message_timestamp(message) -> float:
        """Read a ROS header stamp, falling back to the local wall clock."""
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        try:
            candidate = float(getattr(stamp, "sec", 0.0)) + (
                float(getattr(stamp, "nanosec", 0.0)) / 1e9
            )
        except (TypeError, ValueError):
            candidate = 0.0
        return candidate if math.isfinite(candidate) and candidate > 0.0 else time.time()

    def _format_detections(
        self,
        round_id: int,
        detections: list[DetectionObject],
        object_class: Optional[str] = None,
        coordinate_frame: str = "map",
        snapshot_detections: Optional[list[DetectionObject]] = None,
    ) -> str:
        snapshot = (
            list(snapshot_detections)
            if snapshot_detections is not None
            else list(detections)
        )
        if object_class:
            detections = [
                d for d in detections
                if d.class_name.lower() == object_class.lower()
            ]

        if not detections:
            msg = "当前未检测到任何目标物体。"
            if object_class:
                msg = f"当前未检测到类别为 '{object_class}' 的目标。"
            if snapshot:
                _detection_snapshots.confirm(round_id, snapshot, msg)
            else:
                _detection_snapshots.fail(round_id, msg)
            return msg

        label = f"（过滤: {object_class}）" if object_class else ""
        lines = [
            f"检测到 {len(detections)} 个目标{label}"
            f"（坐标系: {coordinate_frame}，可直接用于 Nav2）:",
            f"方向汇总: {summarize_detection_directions(detections)}",
        ]
        for d in detections:
            display_name = CLASS_NAMES_ZH.get(
                d.class_name.lower(), d.class_name
            )
            lines.append(
                f"  - {display_name}（{d.class_name}）: "
                f"在小车{d.direction}; "
                f"map坐标 x={d.x:.2f}m, y={d.y:.2f}m; "
                f"检测高度 z={d.z:.2f}m; 置信度={d.confidence:.2f}; "
                f"已连续确认 {d.confirmed_hits} 帧"
            )
        result = "\n".join(lines)
        _detection_snapshots.confirm(round_id, snapshot, result)
        return result

    @staticmethod
    def _direction_from_robot_frame(x: float, y: float) -> str:
        """Return a Chinese direction in base_link coordinates."""
        return direction_from_robot_frame(x, y)

    def _get_direction_transform(self, source_frame: str):
        source_frame = self._normalize_frame_id(source_frame)
        if not source_frame:
            return None
        if source_frame == "base_link":
            return True
        try:
            return self.connector.get_transform(
                target_frame="base_link",
                source_frame=source_frame,
                timeout_sec=1.0,
            )
        except Exception as exc:
            logger.warning("无法计算相对方向 %s→base_link: %s", source_frame, exc)
            return None

    def _apply_direction_transform(
        self, detections: list[DetectionObject], transform
    ) -> list[DetectionObject]:
        if transform is True:
            return [
                d.model_copy(update={
                    "direction": self._direction_from_robot_frame(d.x, d.y)
                })
                for d in detections
            ]
        if transform is None:
            return [
                d.model_copy(update={
                    "direction": d.direction or "方向未知"
                })
                for d in detections
            ]
        q = transform.transform.rotation
        t = transform.transform.translation
        result = []
        for detection in detections:
            x, y, _ = self._apply_transform(
                detection.x, detection.y, detection.z,
                q.x, q.y, q.z, q.w,
                t.x, t.y, t.z,
            )
            result.append(
                detection.model_copy(
                    update={"direction": self._direction_from_robot_frame(x, y)}
                )
            )
        return result

    def _stabilize_socket_payload(
        self,
        client: DetectBBox3DSocketClient,
        start: float,
    ) -> tuple[dict | None, list[DetectionObject]]:
        stabilizer = DetectionStabilizer(
            min_hits=self.confirmation_hits,
            window_size=self.confirmation_window,
            window_seconds=min(self.confirmation_window_seconds, self.timeout_sec),
            match_distance=self.confirmation_distance,
            max_missed_frames=self.confirmation_max_missed_frames,
        )
        last_sequence = None
        latest_payload = None
        while time.time() - start < self.timeout_sec:
            get_with_sequence = getattr(client, "get_latest_with_sequence", None)
            if get_with_sequence is not None:
                latest = get_with_sequence(max_age=self.cache_max_age)
                if latest is None:
                    payload = None
                    receive_sequence = None
                    received_at = None
                else:
                    payload, receive_sequence, received_at = latest
            else:
                payload = client.get_latest(max_age=self.cache_max_age)
                receive_sequence = None
                received_at = None
            if payload is None or not isinstance(payload, dict):
                time.sleep(0.05)
                continue
            if receive_sequence is not None:
                sequence = ("received", receive_sequence)
            else:
                stamp = payload.get("stamp", {}) or {}
                sequence = (
                    stamp.get("sec"),
                    stamp.get("nanosec"),
                    payload.get("frame_sequence"),
                )
            if sequence == last_sequence:
                time.sleep(0.05)
                continue
            last_sequence = sequence
            latest_payload = payload
            detections = self._parse_socket_payload(payload)
            if payload.get("stabilized") and detections:
                return payload, detections
            payload_time = self._payload_timestamp(payload, received_at)
            confirmed = stabilizer.update(detections, payload_time)
            if confirmed:
                return payload, confirmed
            time.sleep(0.05)
        return latest_payload, []

    @staticmethod
    def _normalize_frame_id(frame_id: object) -> str:
        return str(frame_id or "").strip().lstrip("/")

    def _target_frame_name(self) -> str:
        return self._normalize_frame_id(self.target_frame) or "map"

    def _format_transform_error(
        self, source_frame: str, error: Exception
    ) -> str:
        source = self._normalize_frame_id(source_frame) or "未知"
        target = self._target_frame_name()
        logger.error(
            f"检测坐标无法从 {source} 转换到 {target}: {error}"
        )
        return (
            f"检测结果当前属于 {source} 坐标系，未能转换为 {target} 坐标系。"
            "本次不返回坐标，避免把激光雷达坐标误当成 Nav2 目标。"
            f"请确认 AMCL 已完成初始定位，并检查: "
            f"ros2 run tf2_ros tf2_echo {target} {source}"
        )

    @staticmethod
    def _apply_transform(
        px: float, py: float, pz: float,
        qx: float, qy: float, qz: float, qw: float,
        tx: float, ty: float, tz: float,
    ) -> tuple[float, float, float]:
        """对单个点施加旋转 + 平移 (无外部依赖)"""
        # r × v
        rx_cv = qy * pz - qz * py
        ry_cv = qz * px - qx * pz
        rz_cv = qx * py - qy * px
        # r × v + w*v
        ax, ay, az = rx_cv + qw * px, ry_cv + qw * py, rz_cv + qw * pz
        # r × (r × v + w*v)
        bx = qy * az - qz * ay
        by = qz * ax - qx * az
        bz = qx * ay - qy * ax
        # v + 2*b + translation
        return px + 2 * bx + tx, py + 2 * by + ty, pz + 2 * bz + tz

    def _transform_detections(
        self, detections: list[DetectionObject], source_frame: str
    ) -> list[DetectionObject]:
        """把检测坐标从 source_frame 变换到 target_frame"""
        source_frame = self._normalize_frame_id(source_frame)
        target_frame = self._target_frame_name()
        if not source_frame:
            raise DetectionTransformError("检测消息缺少 header.frame_id")

        if source_frame == target_frame:
            return detections  # 同坐标系，不用变

        try:
            tf = self.connector.get_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                timeout_sec=3.0,
            )
        except Exception as e:
            logger.warning(
                f"TF 变换 {source_frame}→{target_frame} 失败: {e}"
            )
            raise DetectionTransformError(
                f"TF 变换 {source_frame}→{target_frame} 失败"
            ) from e

        q = tf.transform.rotation
        t = tf.transform.translation

        result = []
        for det in detections:
            nx, ny, nz = self._apply_transform(
                det.x, det.y, det.z,
                q.x, q.y, q.z, q.w,
                t.x, t.y, t.z,
            )
            result.append(DetectionObject(
                class_name=det.class_name,
                x=nx, y=ny, z=nz,
                size_x=det.size_x,
                size_y=det.size_y,
                confidence=det.confidence,
                direction=det.direction,
                confirmed_hits=det.confirmed_hits,
            ))
        logger.info(
            f"TF: {len(result)} 个目标 {source_frame}→{target_frame}"
        )
        return result

    def _run_detection(
        self,
        round_id: int,
        object_class: Optional[str] = None,
    ) -> str:
        if self.detection_source == "socket":
            client = _get_socket_client(self.socket_host, self.socket_port)

            start = time.time()
            payload, detections = self._stabilize_socket_payload(client, start)

            if payload is None:
                return (
                    f"未收到 socket 检测结果（等待 {self.timeout_sec}s）。"
                    f"请确认 Orin 上已运行 detect_bbox3d_socket_bridge，"
                    f"并监听 {self.socket_host}:{self.socket_port}。"
                )

            if payload.get("stabilized") and payload.get("stabilization_state") != "confirmed":
                return (
                    f"暂未确认稳定目标。已等待 {self.timeout_sec:.1f}s，至少需要"
                    f" {self.confirmation_hits} 帧一致检测。"
                )

            if not detections:
                if object_class:
                    return (
                        f"暂未确认类别为 '{object_class}' 的稳定目标。"
                        f"已等待 {self.timeout_sec:.1f}s，至少需要"
                        f" {self.confirmation_hits} 帧一致检测。"
                    )
                return (
                    f"暂未确认稳定目标。已等待 {self.timeout_sec:.1f}s，至少需要"
                    f" {self.confirmation_hits} 帧一致检测。"
                )
            source_frame = self._normalize_frame_id(payload.get("frame_id", ""))
            direction_transform = self._get_direction_transform(source_frame)
            detections = self._apply_direction_transform(
                detections, direction_transform
            )
            try:
                detections = self._transform_detections(detections, source_frame)
            except DetectionTransformError as e:
                return self._format_transform_error(source_frame, e)
            all_detections = list(detections)
            return self._format_detections(
                round_id,
                detections,
                object_class,
                coordinate_frame=self._target_frame_name(),
                snapshot_detections=all_detections,
            )

        # ROS 直连模式也使用同一套多帧确认逻辑，避免 socket 和 ROS
        # 两条入口对“已确认目标”的定义不一致。
        import rclpy
        from vision_msgs.msg import Detection3DArray

        cache = []
        node = rclpy.create_node(f"rai_detect_{int(time.time())}")

        def cb(msg): cache.append(msg)
        node.create_subscription(Detection3DArray, self.topic, cb, 10)

        stabilizer = DetectionStabilizer(
            min_hits=self.confirmation_hits,
            window_size=self.confirmation_window,
            window_seconds=min(self.confirmation_window_seconds, self.timeout_sec),
            match_distance=self.confirmation_distance,
            max_missed_frames=self.confirmation_max_missed_frames,
        )
        start = time.time()
        payload = None
        detections = []
        source_frame = None
        while time.time() - start < self.timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.2)
            while cache:
                msg = cache.pop(0)
                current_frame = self._normalize_frame_id(
                    getattr(getattr(msg, "header", None), "frame_id", "")
                )
                if source_frame is not None and current_frame != source_frame:
                    # 不把不同坐标系的点混进同一条轨迹。
                    stabilizer = DetectionStabilizer(
                        min_hits=self.confirmation_hits,
                        window_size=self.confirmation_window,
                        window_seconds=min(
                            self.confirmation_window_seconds,
                            self.timeout_sec,
                        ),
                        match_distance=self.confirmation_distance,
                        max_missed_frames=self.confirmation_max_missed_frames,
                    )
                source_frame = current_frame
                payload = msg
                raw_detections = self._parse_detection3d_array(msg)
                detections = stabilizer.update(
                    raw_detections,
                    self._message_timestamp(msg),
                )
                if detections:
                    break
            if detections:
                break

        node.destroy_node()

        if payload is None:
            return (
                f"未收到检测结果（等待 {self.timeout_sec}s，"
                "没有收到有效检测帧）。"
                f"请确认 Orin 的 VoteNet 正在发布 {self.topic}。"
            )

        if not detections:
            return (
                f"暂未确认稳定目标。已等待 {self.timeout_sec:.1f}s，至少需要"
                f" {self.confirmation_hits} 帧类别和位置一致的检测。"
            )

        # 先计算相对方向，再把已确认目标变换到 map。
        direction_transform = self._get_direction_transform(source_frame)
        detections = self._apply_direction_transform(
            detections, direction_transform
        )
        try:
            detections = self._transform_detections(detections, source_frame)
        except DetectionTransformError as e:
            return self._format_transform_error(source_frame, e)

        return self._format_detections(
            round_id,
            detections,
            object_class,
            coordinate_frame=self._target_frame_name(),
            snapshot_detections=list(detections),
        )

    def _run(self, object_class: Optional[str] = None) -> str:
        round_id = _detection_snapshots.begin()
        try:
            result = self._run_detection(round_id, object_class)
        except Exception as exc:
            _detection_snapshots.fail(round_id, f"检测执行异常：{exc}")
            raise

        state = _detection_snapshots.read()
        if state["round_id"] == round_id and state["status"] == "pending":
            _detection_snapshots.fail(round_id, result)
        return result


class NavigateToDetectedTargetInput(BaseModel):
    target: str = Field(
        default="",
        description=(
            "刚才检测结果中的目标描述。优先使用方向加类别，如"
            "'前方偏左的椅子'、'右侧的柜子'；也可只填类别。"
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_target(cls, values):
        """Keep older model calls useful without adding tool parameters."""
        if not isinstance(values, dict):
            return values
        values = dict(values)
        if not str(values.get("target") or "").strip():
            for key in ("name", "object", "object_class", "class_name"):
                candidate = values.get(key)
                if candidate is not None and str(candidate).strip():
                    values["target"] = candidate
                    break
        return values


class NavigateToDetectedTargetTool(BaseTool):
    """Navigate using the last confirmed detection snapshot, never a new frame."""

    name: str = "navigate_to_detected_target"
    description: str = (
        "目标导航必须首先调用本工具。它只使用最近一次 get_detections "
        "保存的已确认快照，不会为已有快照重新检测。用户说'去左侧的椅子'、"
        "'去前方偏右的椅子'、'找桌子'时直接使用；从未执行过检测时，"
        "工具会自动调用一次 get_detections。最近一轮失败或无匹配目标时不得"
        "回退旧轮次。工具会导航到物体前的安全接近点，不会把障碍物中心直接"
        "作为 Nav2 终点。"
    )
    args_schema: Type[NavigateToDetectedTargetInput] = NavigateToDetectedTargetInput
    navigate_tool: object = Field(..., exclude=True)
    detection_tool: object | None = Field(default=None, exclude=True)
    costmap_service: str = Field(
        default="/global_costmap/get_costmap",
        exclude=True,
    )
    costmap_timeout_sec: float = Field(default=2.0, exclude=True)

    def _load_global_costmap(self) -> dict:
        response = self.navigate_tool.connector.service_call(
            ROS2Message(payload={}),
            target=self.costmap_service,
            msg_type="nav2_msgs/srv/GetCostmap",
            timeout_sec=self.costmap_timeout_sec,
        )
        payload = response.payload
        costmap_message = (
            payload.get("map")
            if isinstance(payload, dict)
            else getattr(payload, "map", None)
        )
        if costmap_message is None:
            raise ValueError("global costmap 响应缺少 map 字段")
        return normalize_costmap(costmap_message)

    def _run(self, target: str = "") -> str:
        target = str(target or "").strip()
        if not target:
            return "请说明要导航的目标类别或方向，例如“去前方偏左的椅子”。"

        latest_round = get_latest_detection_round()
        if latest_round["status"] == "never":
            if self.detection_tool is not None:
                try:
                    # 由工具本身兜底，保证命令行 Agent 也遵守“无快照先检测一次”。
                    self.detection_tool.invoke({"object_class": None})
                except Exception as exc:
                    logger.warning("自动获取检测结果失败: %s", exc)
                latest_round = get_latest_detection_round()
            if latest_round["status"] == "never":
                return "没有任何检测轮次，自动检测未返回结果，请先查看检测结果。"
        if latest_round["status"] != "confirmed":
            detail = latest_round["message"] or "本轮未形成已确认目标"
            return f"最近一轮检测没有可用的结构化快照：{detail}"
        detections = latest_round["detections"]

        selected = select_detection_targets(detections, target)
        if len(selected) != 1:
            if not selected:
                return f"最近一轮检测中没有匹配“{target}”的目标。"
            choices = []
            for item in selected:
                display_name = CLASS_NAMES_ZH.get(
                    item.class_name.lower(), item.class_name
                )
                choice = f"{item.direction}的{display_name}"
                if choice not in choices:
                    choices.append(choice)
            choices_text = "、".join(choices)
            if len(choices) == 1:
                return (
                    f"检测快照中有多个“{choices_text}”，当前方向仍无法安全区分。"
                    "请让小车稍微改变朝向后重新检测，不需要选择序号。"
                )
            return (
                f"检测快照中有多个目标匹配“{target}”：{choices_text}。"
                f"请直接按方向说明，例如“去{choices[0]}”，不需要选择序号。"
            )

        detection = selected[0]
        try:
            localization = self.navigate_tool.localization_manager.require_localized()
            confirmed_pose = localization.get("pose")
            if not confirmed_pose:
                raise LocalizationError("没有已确认的 AMCL 位姿")
            standoff_distance = calculate_detection_standoff(
                detection.size_x,
                detection.size_y,
            )
            candidates, target_distance = generate_standoff_candidates(
                confirmed_pose,
                detection.x,
                detection.y,
                standoff_distance,
            )
        except (AttributeError, KeyError, TypeError, ValueError, LocalizationError) as exc:
            return f"无法计算目标安全接近点：{exc}。"

        class_name = CLASS_NAMES_ZH.get(
            detection.class_name.lower(), detection.class_name
        )
        if target_distance <= standoff_distance:
            return (
                f"刚才确认的{class_name}位于小车{detection.direction}，"
                f"目标中心 map 坐标 x={detection.x:.2f}m, y={detection.y:.2f}m；"
                f"当前距离约 {target_distance:.2f}m，已在 "
                f"{standoff_distance:.2f}m 安全接近范围内，未启动导航。"
            )

        approach_x, approach_y = candidates[0]
        approach_note = "按目标近侧计算"
        try:
            costmap = self._load_global_costmap()
        except Exception as exc:
            logger.warning("读取 global costmap 失败，使用近侧接近点: %s", exc)
            approach_note = "global costmap 不可用，已使用目标近侧回退点"
        else:
            selected_candidate = select_costmap_candidate(
                candidates,
                costmap,
                start=(float(confirmed_pose["x"]), float(confirmed_pose["y"])),
            )
            if selected_candidate is None:
                return (
                    f"最近一轮确认的{class_name}位于小车{detection.direction}，"
                    f"但目标中心周围 {standoff_distance:.2f}m 的候选接近点"
                    "均处于障碍、未知区域或地图范围外，未启动导航。"
                )
            approach_x, approach_y, approach_cost = selected_candidate
            approach_note = (
                "已通过 global costmap 障碍及路径连通性筛选"
                f"（代价值 {approach_cost}）"
            )

        result = self.navigate_tool.invoke({
            "x": approach_x,
            "y": approach_y,
        })
        result_text = str(result)
        launch_label = "已使用" if "导航已开始" in result_text else "未能启动"
        return (
            f"{launch_label}刚才确认的{class_name}（小车{detection.direction}）"
            f"规划接近导航。"
            f"物体中心 map 坐标 x={detection.x:.2f}m, y={detection.y:.2f}m；"
            f"安全接近点 x={approach_x:.2f}m, y={approach_y:.2f}m，"
            f"与物体中心保留 {standoff_distance:.2f}m，{approach_note}。\n"
            f"{result_text}"
        )


# ─── Tool: 发送导航目标 ──────────────────────────────────────────────────

class NavigateToCoordinatesToolInput(BaseModel):
    x: float = Field(
        description="map 坐标系中的目标 x 坐标，单位 m。用户直接给出的 x 可直接使用。"
    )
    y: float = Field(
        description="map 坐标系中的目标 y 坐标，单位 m。用户直接给出的 y 可直接使用。"
    )


class NavigateToCoordinatesTool(BaseTool):
    """向 Orin Nav2 发送 map 坐标导航目标。"""

    name: str = "navigate_to_coordinates"
    description: str = (
        "控制小车导航到指定的 map 坐标。"
        "用户明确提供 x、y 时直接调用本工具，不需要先调用 get_detections。"
        "导航前必须已通过页面的自动定位按钮完成 AMCL 定位。"
        "只需要 x(m)、y(m)；若定位使用了 RViz 2D Pose Estimate，目标朝向"
        "沿用该 Pose 选择的 yaw，否则根据当前位置到目标计算朝向。"
        "例如用户说‘去 map 坐标 x=-4.2, y=2.97’，调用 "
        "{x: -4.2, y: 2.97}。"
    )
    args_schema: Type[NavigateToCoordinatesToolInput] = (
        NavigateToCoordinatesToolInput
    )

    connector: ROS2Connector = Field(..., exclude=True)
    localization_manager: object | None = Field(default=None, exclude=True)
    frame_id: str = Field(default="map")
    base_frame: str = Field(default="base_link")
    action_name: str = Field(default="/navigate_to_pose")
    action_timeout_sec: float = Field(
        default=10.0,
        description="等待 Nav2 Action Server 和目标接受的最长时间(秒)",
    )

    def _run(self, x: float, y: float) -> str:
        global _active_navigation_action_id
        values = (x, y)
        if not all(math.isfinite(float(value)) for value in values):
            return "导航失败: x、y 必须是有限数字。"
        if not math.isfinite(self.action_timeout_sec) or self.action_timeout_sec <= 0:
            return "导航失败: Action 等待时间必须大于 0 秒。"

        target = self.action_name.strip()
        if not target.startswith("/"):
            target = "/" + target
        frame_id = self.frame_id.strip() or "map"

        try:
            if self.localization_manager is None:
                raise LocalizationError("定位管理器尚未初始化")
            localization = self.localization_manager.require_localized()
            confirmed_pose = localization.get("pose")
            if not confirmed_pose:
                raise LocalizationError("没有已确认的 AMCL 位姿")
            robot_x = float(confirmed_pose["x"])
            robot_y = float(confirmed_pose["y"])
            if not all(math.isfinite(value) for value in (robot_x, robot_y)):
                raise LocalizationError("已确认的 AMCL 位姿无效")
            yaw, yaw_source = resolve_navigation_yaw(confirmed_pose, x, y)
            quat = _quaternion_from_yaw(yaw)
            goal = {
                "pose": {
                    "header": {
                        "frame_id": frame_id,
                        "stamp": self.connector.node.get_clock().now().to_msg(),
                    },
                    "pose": {
                        # Nav2 是二维导航，检测框的高度不能作为目标高度。
                        "position": {"x": x, "y": y, "z": 0.0},
                        "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                    },
                }
            }

            msg = ROS2Message(payload=goal)
            callback_context = {"action_id": None, "future": None}

            def on_done(future) -> None:
                with _navigation_action_lock:
                    action_id = callback_context["action_id"]
                    if action_id is None:
                        callback_context["future"] = future
                        return
                _handle_navigation_done(action_id, future)

            action_id = self.connector.start_action(
                action_data=msg,
                target=target,
                msg_type="nav2_msgs/action/NavigateToPose",
                timeout_sec=self.action_timeout_sec,
                on_done=on_done,
            )
            with _navigation_action_lock:
                callback_context["action_id"] = action_id
                _active_navigation_action_id = action_id
                _navigation_status.update(
                    status="navigating",
                    action_id=action_id,
                    event_id=action_id,
                    x=float(x),
                    y=float(y),
                    message="正在导航中",
                    result_code=None,
                    updated_at=time.time(),
                )
                pending_future = callback_context.pop("future", None)

            if pending_future is not None:
                _handle_navigation_done(action_id, pending_future)

            if yaw_source == "initialpose":
                heading_message = (
                    "已使用 RViz 2D Pose Estimate 选定的朝向 "
                    f"yaw={yaw:.2f}rad"
                )
            else:
                heading_message = (
                    f"已根据小车当前位置({robot_x:.2f}, {robot_y:.2f})"
                    f"计算朝向 yaw={yaw:.2f}rad"
                )

            return (
                f"导航已开始 (ID: {action_id})。\n"
                f"目标({frame_id}): x={x:.2f}m, y={y:.2f}m\n"
                f"{heading_message}\n"
                f"小车正在导航中，到达后会提示导航完成。"
            )
        except LocalizationError as e:
            message = f"AMCL 定位不可用：{e}"
            _mark_navigation_failed(x, y, message)
            logger.error(message)
            return f"导航未启动：{message}。"
        except Exception as e:
            _mark_navigation_failed(x, y, f"导航失败：{e}")
            logger.error(f"导航失败: {e}")
            return f"导航失败：{e}。"


# ─── Tool: 取消导航 ──────────────────────────────────────────────────────

class CancelNavigationTool(BaseTool):
    """取消导航任务。"""

    name: str = "cancel_navigation"
    description: str = "取消当前导航任务，让小车停止。用户说'停下'/'停止'时使用。"

    connector: ROS2Connector = Field(..., exclude=True)
    localization_manager: object | None = Field(default=None, exclude=True)

    def _run(self) -> str:
        global _active_navigation_action_id
        action_id = None
        try:
            localization_canceled = False
            if self.localization_manager is not None:
                localization_canceled = (
                    self.localization_manager.cancel_global_localization()
                )
            with _navigation_action_lock:
                action_id = _active_navigation_action_id
            if not action_id:
                if localization_canceled:
                    return "自动定位已取消，小车正在停止。"
                return "当前没有可取消的导航任务。"

            # terminate_action expects the goal handle returned by start_action,
            # not the /navigate_to_pose action name.
            _mark_navigation_canceling(action_id)
            self.connector.terminate_action(action_id)
            if localization_canceled:
                return "自动定位和导航取消请求均已发送，小车正在停止。"
            return "取消请求已发送，小车正在停止。"
        except Exception as e:
            if action_id:
                _finish_navigation(action_id, "failed", f"取消导航失败：{e}")
            logger.error(f"取消导航失败: {e}")
            return f"取消失败: {e}"
