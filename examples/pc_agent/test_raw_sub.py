#!/usr/bin/env python3
"""Socket test for Orin /detect_bbox3d bridge."""

import argparse
import json
import socket
import sys


def main(host="127.0.0.1", port=8765, timeout=15.0):
    print(f"等待 detect socket {host}:{port} (最长 {timeout}s)...")

    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except OSError as exc:
        print(f"\n连接失败: {exc}")
        return 1

    sock.settimeout(timeout)
    with sock:
        file_obj = sock.makefile("r", encoding="utf-8")
        try:
            line = file_obj.readline()
        except OSError as exc:
            print(f"\n读取失败: {exc}")
            return 1

    if not line:
        print(f"\n{timeout}s 内未收到 socket 消息")
        return 1

    msg = json.loads(line)
    detections = msg.get("detections", [])
    print(
        f"\n收到消息, frame_id={msg.get('frame_id', '')}, "
        f"detections={len(detections)}"
    )

    for i, det in enumerate(detections, 1):
        center = det.get("center", {})
        print(
            f"    {i}. {det.get('class_id', ''):12s} "
            f"({center.get('x', 0.0):.2f},"
            f"{center.get('y', 0.0):.2f},"
            f"{center.get('z', 0.0):.2f}) "
            f"conf={det.get('score', 0.0):.2f}"
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    sys.exit(main(args.host, args.port, args.timeout))
