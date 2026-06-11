#!/usr/bin/env python3
"""
PC ↔ Orin 通信测试 — 纯 rclpy，不经过 RAI connector

用法:
    python examples/pc_agent/test_connection.py
"""

import sys
import signal
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from vision_msgs.msg import Detection3DArray


def main(topic="/detect_bbox3d", timeout=15.0):
    rclpy.init()

    node = Node("test_detect")

    latest = []
    event = threading.Event()

    def on_msg(msg: Detection3DArray):
        latest.append(msg)
        event.set()

    node.create_subscription(Detection3DArray, topic, on_msg, 10)

    # 用独立线程 spin
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print(f"⏳ 等待 {topic} (最长 {timeout}s)...")

    received = event.wait(timeout=timeout)

    executor.shutdown()
    spin_thread.join(timeout=2)
    node.destroy_node()
    rclpy.shutdown()

    if not received:
        print(f"\n✗ {timeout}s 内未收到消息")
        print(f"  VoteNet 在发布吗？ ros2 topic echo 能收到？")
        return 1

    msg = latest[0]
    print(f"\n✓ 收到消息")
    print(f"  frame_id: {msg.header.frame_id}")
    print(f"  检测数: {len(msg.detections)}")
    for i, det in enumerate(msg.detections):
        if det.results:
            best = det.results[0]
            p = best.pose.pose.position
            print(f"    {i+1}. {best.hypothesis.class_id:10s} "
                  f"({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) "
                  f"conf={best.hypothesis.score:.2f}")
        elif hasattr(det, 'bbox'):
            p = det.bbox.center.position
            print(f"    {i+1}. (无标签) ({p.x:.2f}, {p.y:.2f}, {p.z:.2f})")

    print(f"\n{'='*50}")
    print(f"✓ PC ↔ Orin 通信正常")
    print(f"  坐标系: {msg.header.frame_id}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
