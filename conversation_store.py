# -----------------------------
# Conversation Store (in-memory)
# -----------------------------
from typing import Dict, List
from threading import Lock
import time

class ConversationStore:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._lock = Lock()
        self._data: Dict[str, Dict] = {}

    def get(self, session_id: str) -> List[dict]:
        now = time.time()
        with self._lock:
            item = self._data.get(session_id)
            if not item or (now - item["ts"] > self.ttl):
                self._data[session_id] = {"ts": now, "messages": []}
            self._data[session_id]["ts"] = now
            return self._data[session_id]["messages"]

    def clear(self, session_id: str):
        with self._lock:
            self._data.pop(session_id, None)

conversations = ConversationStore(ttl_seconds=3600)