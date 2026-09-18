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
from pydantic import BaseModel, Field

from rai.communication.ros2 import ROS2Message
from rai.communication.ros2.connectors import ROS2Connector

from .detect_socket_client import DetectBBox3DSocketClient

logger = logging.getLogger(__name__)

# 模块级检测缓存 — 持续订阅，每次 _run 只读缓存
_detection_cache: dict = {}
_detection_lock = Lock()
_socket_clients: dict[tuple[str, int], DetectBBox3DSocketClient] = {}
_socket_clients_lock = Lock()
_detection_snapshot_lock = Lock()
_last_confirmed_detections: list = []
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

_CLASS_NAMES_ZH = {
    "bed": "床",
    "chair": "椅子",
    "sofa": "沙发",
    "table": "桌子",
    "desk": "书桌",
    "cabinet": "柜子",
    "door": "门",
    "window": "窗户",
    "bookshelf": "书架",
    "toilet": "马桶",
    "sink": "水槽",
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
    confidence: float = Field(default=0.0)
    direction: str = Field(default="方向未知")
    confirmed_hits: int = Field(default=1)


def _set_detection_snapshot(detections: list[DetectionObject]) -> None:
    global _last_confirmed_detections
    with _detection_snapshot_lock:
        _last_confirmed_detections = [d.model_copy() for d in detections]


def get_detection_snapshot() -> list[DetectionObject]:
    with _detection_snapshot_lock:
        return [d.model_copy() for d in _last_confirmed_detections]


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
            confidence=sum(d.confidence for d in recent) / len(recent),
            direction=self.last_detection.direction,
            confirmed_hits=len(self.samples),
        )


class DetectionStabilizer:
    """Confirm objects after repeated spatially consistent observations."""

    def __init__(
        self,
        min_hits: int = 3,
        window_size: int = 5,
        window_seconds: float = 1.0,
        match_distance: float = 0.5,
        max_missed_frames: int = 3,
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

        for _, track_index, detection_index in sorted(candidates):
            if detection_index not in unmatched:
                continue
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
    confirmation_window: int = Field(default=5)
    confirmation_distance: float = Field(default=0.5)

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

            x = float(center.get("x", 0.0) or 0.0)
            y = float(center.get("y", 0.0) or 0.0)
            z = float(center.get("z", 0.0) or 0.0)

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
                confidence=score,
                direction=str(
                    det.get("relative_direction", det.get("direction", "方向未知"))
                ),
                confirmed_hits=int(det.get("confirmed_hits", 1) or 1),
            ))
        return detections

    def _format_detections(
        self,
        detections: list[DetectionObject],
        object_class: Optional[str] = None,
        coordinate_frame: str = "map",
        snapshot_detections: Optional[list[DetectionObject]] = None,
    ) -> str:
        snapshot = snapshot_detections or detections
        if object_class:
            detections = [
                d for d in detections
                if d.class_name.lower() == object_class.lower()
            ]

        if not detections:
            msg = "当前未检测到任何目标物体。"
            if object_class:
                msg = f"当前未检测到类别为 '{object_class}' 的目标。"
            return msg

        _set_detection_snapshot(snapshot)

        label = f"（过滤: {object_class}）" if object_class else ""
        lines = [
            f"检测到 {len(detections)} 个目标{label}"
            f"（坐标系: {coordinate_frame}，可直接用于 Nav2）:"
        ]
        for i, d in enumerate(detections, 1):
            display_name = _CLASS_NAMES_ZH.get(
                d.class_name.lower(), d.class_name
            )
            lines.append(
                f"  {i}. {display_name}（{d.class_name}）: "
                f"在小车{d.direction}; "
                f"map坐标 x={d.x:.2f}m, y={d.y:.2f}m; "
                f"检测高度 z={d.z:.2f}m; 置信度={d.confidence:.2f}; "
                f"已连续确认 {d.confirmed_hits} 帧"
            )
        return "\n".join(lines)

    @staticmethod
    def _direction_from_robot_frame(x: float, y: float) -> str:
        """Return a coarse Chinese direction in base_link coordinates."""
        angle = math.atan2(y, x)
        if -math.pi / 8 <= angle < math.pi / 8:
            return "正前方"
        if math.pi / 8 <= angle < 3 * math.pi / 8:
            return "左前方"
        if 3 * math.pi / 8 <= angle < 5 * math.pi / 8:
            return "左侧"
        if 5 * math.pi / 8 <= angle < 7 * math.pi / 8:
            return "左后方"
        if angle >= 7 * math.pi / 8 or angle < -7 * math.pi / 8:
            return "正后方"
        if -7 * math.pi / 8 <= angle < -5 * math.pi / 8:
            return "右后方"
        if -5 * math.pi / 8 <= angle < -3 * math.pi / 8:
            return "右侧"
        return "右前方"

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
            window_seconds=min(1.0, self.timeout_sec),
            match_distance=self.confirmation_distance,
        )
        last_sequence = None
        latest_payload = None
        while time.time() - start < self.timeout_sec:
            payload = client.get_latest(max_age=self.cache_max_age)
            if payload is None:
                time.sleep(0.05)
                continue
            sequence = (payload.get("stamp", {}).get("sec"),
                        payload.get("stamp", {}).get("nanosec"))
            if sequence == last_sequence:
                time.sleep(0.05)
                continue
            last_sequence = sequence
            latest_payload = payload
            detections = self._parse_socket_payload(payload)
            if payload.get("stabilized") and detections:
                return payload, detections
            payload_time = time.time()
            stamp = payload.get("stamp", {}) or {}
            if stamp.get("sec") is not None and stamp.get("nanosec") is not None:
                candidate_time = float(stamp["sec"]) + float(stamp["nanosec"]) / 1e9
                if candidate_time > 0:
                    payload_time = candidate_time
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
                confidence=det.confidence,
                direction=det.direction,
                confirmed_hits=det.confirmed_hits,
            ))
        logger.info(
            f"TF: {len(result)} 个目标 {source_frame}→{target_frame}"
        )
        return result

    def _run(self, object_class: Optional[str] = None) -> str:
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
                detections,
                object_class,
                coordinate_frame=self._target_frame_name(),
                snapshot_detections=all_detections,
            )

        # 直接用 test_raw_sub.py 的模式 — spin_once 循环
        import rclpy
        from vision_msgs.msg import Detection3DArray

        cache = []
        node = rclpy.create_node(f"rai_detect_{int(time.time())}")

        def cb(msg): cache.append(msg)
        node.create_subscription(Detection3DArray, self.topic, cb, 10)

        start = time.time()
        payload = None
        empty_count = 0
        while time.time() - start < self.timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.2)
            if cache:
                msg = cache.pop(0)
                if msg.detections:
                    payload = msg
                    break
                else:
                    empty_count += 1

        node.destroy_node()

        if payload is None:
            return (
                f"未收到检测结果（等待 {self.timeout_sec}s，"
                f"收到 {empty_count} 个空帧）。"
                f"请确认 Orin 的 VoteNet 正在发布 {self.topic}。"
            )

        detections = self._parse_detection3d_array(payload)

        # TF 变换: rslidar → map
        source_frame = self._normalize_frame_id(payload.header.frame_id)
        try:
            detections = self._transform_detections(detections, source_frame)
        except DetectionTransformError as e:
            return self._format_transform_error(source_frame, e)

        # 按类别过滤
        if object_class:
            detections = [
                d for d in detections
                if d.class_name.lower() == object_class.lower()
            ]

        if not detections:
            msg = "当前未检测到任何目标物体。"
            if object_class:
                msg = f"当前未检测到类别为 '{object_class}' 的目标。"
            return msg

        return self._format_detections(
            detections,
            object_class,
            coordinate_frame=self._target_frame_name(),
        )


class NavigateToDetectedTargetInput(BaseModel):
    target: str = Field(
        description=(
            "刚才检测结果中的目标选择。可填序号，如'1'；方向，如'左侧'、"
            "'左前方'；或类别，如'椅子'。"
        )
    )


class NavigateToDetectedTargetTool(BaseTool):
    """Navigate using the last confirmed detection snapshot, never a new frame."""

    name: str = "navigate_to_detected_target"
    description: str = (
        "根据最近一次 get_detections 返回的已确认目标导航。"
        "用户说'去左侧的椅子'、'去第一个目标'时使用本工具；"
        "不要重新调用 get_detections，也不要读取最新检测。"
    )
    args_schema: Type[NavigateToDetectedTargetInput] = NavigateToDetectedTargetInput
    navigate_tool: object = Field(..., exclude=True)

    def _run(self, target: str) -> str:
        detections = get_detection_snapshot()
        if not detections:
            return "没有可用的已确认检测快照，请先调用 get_detections。"

        query = str(target).strip().lower()
        selected = []
        try:
            index = int(query) - 1
            if 0 <= index < len(detections):
                selected = [detections[index]]
        except ValueError:
            pass

        if not selected:
            direction_aliases = {
                "左边": "左侧",
                "左面": "左侧",
                "右边": "右侧",
                "右面": "右侧",
                "前面": "正前方",
                "后面": "正后方",
            }
            normalized_query = direction_aliases.get(query, query)
            selected = [
                detection for detection in detections
                if normalized_query in detection.class_name.lower()
                or normalized_query in detection.direction.lower()
                or query in detection.class_name.lower()
                or query in detection.direction.lower()
            ]
        if len(selected) != 1:
            if not selected:
                return f"最近的检测快照中没有匹配“{target}”的目标。"
            return f"检测快照中有多个目标匹配“{target}”，请指定序号。"

        detection = selected[0]
        result = self.navigate_tool.invoke({
            "x": detection.x,
            "y": detection.y,
        })
        class_name = _CLASS_NAMES_ZH.get(
            detection.class_name.lower(), detection.class_name
        )
        return (
            f"已使用刚才确认的{class_name}（小车{detection.direction}）坐标导航，"
            f"目标 map 坐标 x={detection.x:.2f}m, y={detection.y:.2f}m。\n"
            f"{result}"
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
        "只需要 x(m)、y(m)；工具会根据小车当前位置自动计算到目标的朝向。"
        "例如用户说‘去 map 坐标 x=-4.2, y=2.97’，调用 "
        "{x: -4.2, y: 2.97}。"
    )
    args_schema: Type[NavigateToCoordinatesToolInput] = (
        NavigateToCoordinatesToolInput
    )

    connector: ROS2Connector = Field(..., exclude=True)
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
        base_frame = self.base_frame.strip() or "base_link"

        try:
            robot_tf = self.connector.get_transform(
                target_frame=frame_id,
                source_frame=base_frame,
                timeout_sec=3.0,
            )
            robot_x = float(robot_tf.transform.translation.x)
            robot_y = float(robot_tf.transform.translation.y)
            yaw = math.atan2(y - robot_y, x - robot_x)
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

            return (
                f"导航已开始 (ID: {action_id})。\n"
                f"目标({frame_id}): x={x:.2f}m, y={y:.2f}m\n"
                f"已根据小车当前位置({robot_x:.2f}, {robot_y:.2f})"
                f"计算朝向 yaw={yaw:.2f}rad\n"
                f"小车正在导航中，到达后会提示导航完成。"
            )
        except Exception as e:
            _mark_navigation_failed(x, y, f"导航失败：{e}")
            logger.error(f"导航失败: {e}")
            return f"导航失败: {e}。Orin Nav2 是否运行?"


# ─── Tool: 取消导航 ──────────────────────────────────────────────────────

class CancelNavigationTool(BaseTool):
    """取消导航任务。"""

    name: str = "cancel_navigation"
    description: str = "取消当前导航任务，让小车停止。用户说'停下'/'停止'时使用。"

    connector: ROS2Connector = Field(..., exclude=True)

    def _run(self) -> str:
        global _active_navigation_action_id
        action_id = None
        try:
            with _navigation_action_lock:
                action_id = _active_navigation_action_id
            if not action_id:
                return "当前没有可取消的导航任务。"

            # terminate_action expects the goal handle returned by start_action,
            # not the /navigate_to_pose action name.
            _mark_navigation_canceling(action_id)
            self.connector.terminate_action(action_id)
            return "取消请求已发送，小车正在停止。"
        except Exception as e:
            if action_id:
                _finish_navigation(action_id, "failed", f"取消导航失败：{e}")
            logger.error(f"取消导航失败: {e}")
            return f"取消失败: {e}"
