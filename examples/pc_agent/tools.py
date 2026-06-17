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
import time
from threading import Lock
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tf_transformations import quaternion_from_euler

from rai.communication.ros2 import ROS2Message
from rai.communication.ros2.connectors import ROS2Connector

logger = logging.getLogger(__name__)

# 模块级检测缓存 — 持续订阅，每次 _run 只读缓存
_detection_cache: dict = {}
_detection_lock = Lock()


# ─── Data Models ──────────────────────────────────────────────────────────

class DetectionObject(BaseModel):
    class_name: str = Field(description="类别: bed, chair, table...")
    x: float = Field(description="x (m)")
    y: float = Field(description="y (m)")
    z: float = Field(description="z (m)")
    confidence: float = Field(default=0.0)


# ─── Tool: 获取检测结果 ───────────────────────────────────────────────────

class GetDetectionsToolInput(BaseModel):
    object_class: Optional[str] = Field(
        default=None,
        description="按类别过滤，如 'chair'。不填返回全部。"
    )


class GetDetectionsTool(BaseTool):
    """读取 Orin VoteNet 发布的 3D 室内检测结果。

    使用原生 rclpy subscriber（不经过 RAI 的 receive_message），
    持续监听 /detect_bbox3d，缓存最新消息。每次 LLM 调用时直接读缓存。
    """

    name: str = "get_detections"
    description: str = (
        "获取小车 VoteNet 检测到的周围物体及其 3D 坐标。"
        "可选参数 object_class 按类别过滤（如 'chair'）。"
        "返回物体类别、地图坐标(x,y,z)、置信度。"
    )
    args_schema: Type[GetDetectionsToolInput] = GetDetectionsToolInput

    connector: ROS2Connector = Field(..., exclude=True)
    topic: str = Field(default="/detect_bbox3d")
    target_frame: str = Field(default="map", description="TF 变换目标坐标系")
    cache_max_age: float = Field(default=10.0, description="缓存有效时间(秒)")
    timeout_sec: float = Field(default=15.0)

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
        import math

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
        if source_frame == self.target_frame:
            return detections  # 同坐标系，不用变

        try:
            tf = self.connector.get_transform(
                target_frame=self.target_frame,
                source_frame=source_frame,
                timeout_sec=3.0,
            )
        except Exception as e:
            logger.warning(
                f"TF 变换 {source_frame}→{self.target_frame} 失败: {e}，"
                f"返回原始坐标"
            )
            return detections

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
            ))
        logger.info(
            f"TF: {len(result)} 个目标 {source_frame}→{self.target_frame}"
        )
        return result

    def _run(self, object_class: Optional[str] = None) -> str:
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
        source_frame = payload.header.frame_id
        detections = self._transform_detections(detections, source_frame)

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

        label = f"（过滤: {object_class}）" if object_class else ""
        lines = [f"检测到 {len(detections)} 个目标{label}:"]
        for i, d in enumerate(detections, 1):
            lines.append(
                f"  {i}. {d.class_name} "
                f"坐标({d.x:.2f}, {d.y:.2f}, {d.z:.2f}) "
                f"置信度={d.confidence:.2f}"
            )
        return "\n".join(lines)


# ─── Tool: 发送导航目标 ──────────────────────────────────────────────────

class NavigateToCoordinatesToolInput(BaseModel):
    x: float = Field(description="目标 x 坐标 (m)")
    y: float = Field(description="目标 y 坐标 (m)")
    z: float = Field(default=0.0, description="目标 z 坐标 (m)")
    yaw: float = Field(default=0.0, description="朝向 (rad), 0=正前方")


class NavigateToCoordinatesTool(BaseTool):
    """向 Orin Nav2 发送导航目标。"""

    name: str = "navigate_to_coordinates"
    description: str = (
        "控制小车导航到指定地图坐标。"
        "参数: x(前/m), y(左/m), z(高/m), yaw(朝向/rad)。"
        "先通过 get_detections 获取目标坐标，再用此工具导航。"
    )
    args_schema: Type[NavigateToCoordinatesToolInput] = (
        NavigateToCoordinatesToolInput
    )

    connector: ROS2Connector = Field(..., exclude=True)
    frame_id: str = Field(default="map")
    action_name: str = Field(default="/navigate_to_pose")

    def _run(self, x: float, y: float, z: float = 0.0, yaw: float = 0.0) -> str:
        target = self.action_name
        if not target.startswith("/"):
            target = "/" + target

        try:
            quat = quaternion_from_euler(0, 0, yaw)
            goal = {
                "pose": {
                    "header": {
                        "frame_id": self.frame_id,
                        "stamp": self.connector.node.get_clock().now().to_msg(),
                    },
                    "pose": {
                        "position": {"x": x, "y": y, "z": z},
                        "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
                    },
                }
            }

            msg = ROS2Message(payload=goal)
            action_id = self.connector.start_action(
                action_data=msg,
                target=target,
                msg_type="nav2_msgs/action/NavigateToPose",
            )

            return (
                f"导航指令已发送 (ID: {action_id})。\n"
                f"目标: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m, yaw={yaw:.2f}rad\n"
                f"小车正在前往目标..."
            )
        except Exception as e:
            logger.error(f"导航失败: {e}")
            return f"导航失败: {e}。Orin Nav2 是否运行?"


# ─── Tool: 取消导航 ──────────────────────────────────────────────────────

class CancelNavigationTool(BaseTool):
    """取消导航任务。"""

    name: str = "cancel_navigation"
    description: str = "取消当前导航任务，让小车停止。用户说'停下'/'停止'时使用。"

    connector: ROS2Connector = Field(..., exclude=True)

    def _run(self) -> str:
        try:
            actions = self.connector.get_actions_names_and_types()
            cancelled = False
            for name, _ in actions:
                if "navigate_to_pose" in name:
                    self.connector.terminate_action(name)
                    cancelled = True
            return "导航已取消，小车停止。" if cancelled else "当前无导航任务。"
        except Exception as e:
            logger.error(f"取消导航失败: {e}")
            return f"取消失败: {e}"
