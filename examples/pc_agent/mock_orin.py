#!/usr/bin/env python3
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
模拟 Orin 的 3D 室内目标检测发布节点 — 用于 PC Agent 开发测试

发布格式: vision_msgs/Detection3DArray — 与真实 Orin YOLO 输出一致

用法:
    python examples/pc_agent/mock_orin.py [--rate 2.0]

话题:
    发布: /detect_bbox3d (vision_msgs/msg/Detection3DArray)
"""

import argparse
import random
import signal
import sys
import time

import rclpy
from rclpy.node import Node

# ROS 2 标准消息类型 — 与 Orin 实际检测输出一致
from std_msgs.msg import Header
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from geometry_msgs.msg import Pose, PoseWithCovariance, Point, Vector3


# 预定义的模拟场景 — 模拟一个典型室内环境
MOCK_SCENE = [
    ("chair",  1.5, 2.0, 0.0, 0.95),
    ("chair",  1.8, 2.3, 0.0, 0.82),
    ("table",  3.0, 0.5, 0.0, 0.91),
    ("person", 2.5, -1.0, 0.0, 0.88),
    ("person", -1.0, 3.0, 0.0, 0.76),
    ("couch", -2.0, -1.5, 0.0, 0.85),
    ("tv",    -2.5, 0.0, 0.8, 0.93),
]


class MockOrinNode(Node):
    """模拟 Orin 3D 检测节点，发布 vision_msgs/Detection3DArray"""

    def __init__(self, publish_rate: float = 2.0):
        super().__init__("mock_orin_detection")
        self.publisher = self.create_publisher(
            Detection3DArray, "/detect_bbox3d", 10
        )
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_detection)
        self.get_logger().info(
            f"模拟 Orin 3D 检测节点已启动\n"
            f"  话题: /detect_bbox3d\n"
            f"  类型: vision_msgs/Detection3DArray\n"
            f"  频率: {publish_rate}Hz\n"
            f"  模拟物体: {', '.join(set(c for c, *_ in MOCK_SCENE))}"
        )

    def publish_detection(self):
        msg = Detection3DArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for class_name, cx, cy, cz, conf in MOCK_SCENE:
            det = Detection3D()

            # results: 包含类别 + 置信度 + 位姿
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = class_name
            result.hypothesis.score = conf * random.uniform(0.95, 1.0)
            result.pose = PoseWithCovariance()
            result.pose.pose = Pose(
                position=Point(
                    x=cx + random.uniform(-0.03, 0.03),
                    y=cy + random.uniform(-0.03, 0.03),
                    z=cz,
                ),
            )

            # bbox: 3D 包围盒
            det.bbox = BoundingBox3D(
                center=Pose(
                    position=Point(
                        x=cx + random.uniform(-0.03, 0.03),
                        y=cy + random.uniform(-0.03, 0.03),
                        z=cz,
                    ),
                ),
                size=Vector3(
                    x=random.uniform(0.3, 0.6),
                    y=random.uniform(0.3, 0.6),
                    z=random.uniform(0.5, 1.2),
                ),
            )

            det.results = [result]
            msg.detections.append(det)

        self.publisher.publish(msg)
        self.get_logger().debug(f"发布: {len(msg.detections)} 个目标")


def main():
    rclpy.init()

    parser = argparse.ArgumentParser(description="模拟 Orin 3D 检测节点")
    parser.add_argument("--rate", type=float, default=2.0, help="发布频率 Hz")
    args = parser.parse_args()

    node = MockOrinNode(publish_rate=args.rate)

    def shutdown(sig=None, frame=None):
        node.get_logger().info("关闭模拟 Orin 节点...")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"\n🤖 模拟 Orin 3D 检测节点运行中")
    print(f"   消息类型: vision_msgs/Detection3DArray")
    print(f"   话题: /detect_bbox3d")
    print(f"   模拟物体: 2×椅子, 1×桌子, 2×人, 1×沙发, 1×电视\n")
    print(f"   按 Ctrl+C 退出\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
