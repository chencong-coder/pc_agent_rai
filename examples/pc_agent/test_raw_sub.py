#!/usr/bin/env python3
"""
最简订阅测试 — 不经过 RAI，直接用 rclpy 订阅
"""

import sys
import signal
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray


msg_count = 0

def callback(msg: Detection3DArray):
    global msg_count
    msg_count += 1
    print(f"\n收到消息 #{msg_count}")
    print(f"  frame_id: {msg.header.frame_id}")
    print(f"  detections: {len(msg.detections)} 个")
    for i, det in enumerate(msg.detections):
        if det.results:
            cls_name = det.results[0].hypothesis.class_id
            score = det.results[0].hypothesis.score
            pos = det.results[0].pose.pose.position
            print(f"    {i+1}. {cls_name} ({pos.x:.2f},{pos.y:.2f},{pos.z:.2f}) conf={score:.2f}")
        elif hasattr(det, 'bbox'):
            pos = det.bbox.center.position
            print(f"    {i+1}. (无标签) ({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})")


def main():
    rclpy.init()
    node = Node("test_detect_sub", allow_undeclared_parameters=True)
    sub = node.create_subscription(
        Detection3DArray,
        "/detect_bbox3d",
        callback,
        10,
    )
    print("⏳ 等待 /detect_bbox3d 消息...(Ctrl+C 退出)")
    print(f"  发布频率约 0.4Hz，最长等 {2.5*3:.0f}s\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if msg_count == 0:
            print("\n✗ 未收到任何消息")
        else:
            print(f"\n✓ 共收到 {msg_count} 条消息")


if __name__ == "__main__":
    main()
