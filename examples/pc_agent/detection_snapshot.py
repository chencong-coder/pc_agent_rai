"""Thread-safe storage for exactly one latest detection round."""

from copy import deepcopy
from threading import Lock
import time


class DetectionSnapshotStore:
    """Keep the latest detection round and invalidate all older rounds."""

    def __init__(self):
        self._lock = Lock()
        self._next_round_id = 1
        self._round_id = None
        self._status = "never"
        self._detections = []
        self._message = ""
        self._updated_at = 0.0

    def begin(self) -> int:
        """Start a round and immediately make the previous snapshot unusable."""
        with self._lock:
            round_id = self._next_round_id
            self._next_round_id += 1
            self._round_id = round_id
            self._status = "pending"
            self._detections = []
            self._message = ""
            self._updated_at = time.time()
            return round_id

    def confirm(self, round_id: int, detections: list, message: str = "") -> bool:
        """Save a confirmed snapshot if this is still the latest round."""
        with self._lock:
            if self._round_id != round_id:
                return False
            self._status = "confirmed"
            self._detections = deepcopy(list(detections))
            self._message = str(message or "")
            self._updated_at = time.time()
            return True

    def fail(self, round_id: int, message: str) -> bool:
        """Record an unusable latest round without restoring older detections."""
        with self._lock:
            if self._round_id != round_id:
                return False
            self._status = "failed"
            self._detections = []
            self._message = str(message or "")
            self._updated_at = time.time()
            return True

    def read(self) -> dict:
        with self._lock:
            return {
                "round_id": self._round_id,
                "status": self._status,
                "detections": deepcopy(self._detections),
                "message": self._message,
                "updated_at": self._updated_at,
            }

    def reset(self) -> None:
        with self._lock:
            self._round_id = None
            self._status = "never"
            self._detections = []
            self._message = ""
            self._updated_at = 0.0
