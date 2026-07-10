import json
import logging
import socket
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class DetectBBox3DSocketClient:
    """Background TCP client for Orin /detect_bbox3d JSON bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.latest_msg: dict[str, Any] | None = None
        self.latest_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_latest(self, max_age: float = 2.0) -> dict[str, Any] | None:
        with self._lock:
            if self.latest_msg is None:
                return None
            if time.time() - self.latest_time > max_age:
                return None
            return self.latest_msg

    def _loop(self):
        while self._running:
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    logger.info(
                        "已连接检测 socket: %s:%s", self.host, self.port
                    )
                    file_obj = sock.makefile("r", encoding="utf-8")
                    for line in file_obj:
                        if not self._running:
                            return
                        line = line.strip()
                        if not line:
                            continue
                        msg = json.loads(line)
                        with self._lock:
                            self.latest_msg = msg
                            self.latest_time = time.time()
            except Exception as exc:
                if self._running:
                    logger.warning(
                        "检测 socket 断开: %s，1s 后重连", exc
                    )
                    time.sleep(1.0)
