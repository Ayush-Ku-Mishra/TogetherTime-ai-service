# ai-service/memory/rolling_memory.py
import redis
import json
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class RollingMemory:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            # Upstash uses rediss:// (TLS)
            self.redis = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                ssl_cert_reqs=None  # Required for Upstash TLS
            )
            # Test connection
            self.redis.ping()
            self.enabled = True
            print("✅ Redis (Upstash) connected")
        except Exception as e:
            print(f"⚠️ Redis not available: {e} — Memory will use in-process cache")
            self.redis = None
            self.enabled = False
            self._local_cache = {}

    def add_scene_summary(self, roomId: str, summary: dict):
        """
        Store scene with video_id separation.
        Key: room:{roomId}:{videoId}:memory
        """
        video_id = summary.get("video_id", "unknown")

        if self.enabled:
            try:
                key = f"room:{roomId}:{video_id}:memory"
                history = self._get_history_by_key(key)
                history.append(summary)
                if len(history) > 50:
                    history = history[-50:]
                self.redis.set(key, json.dumps(history), ex=86400)
            except Exception as e:
                print(f"⚠️ Redis write failed: {e}")
                self._fallback_write(roomId, summary)
        else:
            self._fallback_write(roomId, summary)

    def get_history_for_video(
        self,
        roomId:   str,
        video_id: str,
    ) -> List[dict]:
        """Get scenes for a specific video only."""

        if self.enabled:
            try:
                key  = f"room:{roomId}:{video_id}:memory"
                data = self.redis.get(key)
                return json.loads(data) if data else []
            except Exception as e:
                print(f"⚠️ Redis read failed: {e}")
                return []
        return []

    def get_history(self, roomId: str) -> List[dict]:
        """
        Get ALL scenes for a room across ALL videos.
        Merges all video keys and sorts by timestamp.
        """
        if self.enabled:
            try:
                # Find all keys for this room
                pattern = f"room:{roomId}:*:memory"
                keys    = self.redis.keys(pattern)

                all_scenes = []
                for key in keys:
                    data = self.redis.get(key)
                    if data:
                        all_scenes.extend(json.loads(data))

                # Sort by timestamp
                all_scenes.sort(key=lambda x: x.get("timestamp", 0))
                return all_scenes

            except Exception as e:
                print(f"⚠️ Redis read failed: {e}")
                return self._local_cache.get(roomId, [])

        return self._local_cache.get(roomId, [])

    def clear(self, roomId: str):
        """Clear ALL video keys for a room."""
        if self.enabled:
            try:
                pattern = f"room:{roomId}:*:memory"
                keys    = self.redis.keys(pattern)
                if keys:
                    self.redis.delete(*keys)
            except Exception:
                pass

        if hasattr(self, '_local_cache'):
            self._local_cache.pop(roomId, None)

    def _get_history_by_key(self, key: str) -> List[dict]:
        """Internal: get history by exact Redis key."""
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else []
        except Exception:
            return []

    def _fallback_write(self, roomId: str, summary: dict):
        if not hasattr(self, '_local_cache'):
            self._local_cache = {}
        if roomId not in self._local_cache:
            self._local_cache[roomId] = []
        self._local_cache[roomId].append(summary)
        if len(self._local_cache[roomId]) > 50:
            self._local_cache[roomId] = self._local_cache[roomId][-50:]