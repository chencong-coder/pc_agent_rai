#!/usr/bin/env python3
"""带 QoS 匹配的订阅测试"""
import sys, time, threading
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection3DArray


def main(topic="/detect_bbox3d", timeout=15.0):
    rclpy.init()
    node = Node("test_detect")

    latest = []
    event = threading.Event()

    # 先试 RELIABLE
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

    def cb(msg):
        latest.append(msg)
        event.set()

    node.create_subscription(Detection3DArray, topic, cb, qos)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()

    print(f"⏳ 等待 {topic} (最长 {timeout}s, QoS=BEST_EFFORT)...")
    ok = event.wait(timeout)

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

    if not ok:
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
