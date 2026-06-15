#!/usr/bin/env python3
"""订阅测试 — 用 rclpy 全局 spin"""
import sys, time
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray


def main(topic="/detect_bbox3d", timeout=15.0):
    rclpy.init()
    node = Node("test_detect")

    latest = []

    def cb(msg):
        latest.append(msg)

    node.create_subscription(Detection3DArray, topic, cb, 10)
    print(f"⏳ 等待 {topic} (最长 {timeout}s)...")

    start = time.time()
    while not latest and time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()

    if not latest:
        print(f"\n✗ {timeout}s 内未收到消息")
        return 1

    msg = latest[0]
    print(f"\n✓ 收到消息, frame_id={msg.header.frame_id}, detections={len(msg.detections)}")
    for i, det in enumerate(msg.detections):
        if det.results:
            b = det.results[0]
            p = b.pose.pose.position
            print(f"    {i+1}. {b.hypothesis.class_id:10s} ({p.x:.2f},{p.y:.2f},{p.z:.2f}) conf={b.hypothesis.score:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
