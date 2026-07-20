"""
State Persistence Service

Saves and restores critical processing state to Redis.
This ensures that after a server restart or page refresh,
the pipeline knows what has already been processed.

What is persisted
-----------------
✅ processing_queue state   (processed_until, current_time, total_duration)
✅ processed_videos         (which video each room has fully processed)
✅ realtime_memory          (raw transcripts)
✅ video_metadata cache     (title, channel, duration)

What is NOT persisted
---------------------
❌ caption_store            (frontend pushes fresh on reconnect)
❌ conversation_memory      (fresh conversation each session is fine)
❌ audio clips              (already on disk)

TTL
---
All Redis keys expire after 24 hours automatically.
No manual cleanup needed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 24 hours in seconds
TTL_SECONDS = 86400


class StatePersistence:

    def __init__(self):
        self._redis = None
        self._enabled = False
        self._local_cache: Dict = {}
        self._connect()

    def _connect(self):
        try:
            import redis
            import os
            from dotenv import load_dotenv
            load_dotenv()

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis = redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=10,
                ssl_cert_reqs=None,
            )
            self._redis.ping()
            self._enabled = True
            logger.info("✅ StatePersistence connected to Redis")

        except Exception as e:
            logger.warning(
                "StatePersistence: Redis unavailable — "
                "using in-memory fallback: %s", e
            )
            self._enabled = False

    # =========================================================================
    # QUEUE STATE
    # =========================================================================

    def save_queue_state(
        self,
        room_id:          str,
        video_id:         str,
        state:            Dict,
    ) -> None:
        """
        Save processing queue state for a room/video.

        Saves only the essential fields — not the queue list itself
        (queue is rebuilt from processed_until on restore).

        Saved fields
        ------------
        - total_duration_s
        - processed_until_s
        - current_time
        - jump_requested
        - jump_target
        """

        key = f"queue_state:{room_id}:{video_id}"

        payload = {
            "room_id":           room_id,
            "video_id":          video_id,
            "total_duration_s":  state.get("total_duration_s",  0.0),
            "processed_until_s": state.get("processed_until_s", 0.0),
            "current_time":      state.get("current_time",      0.0),
            "jump_requested":    state.get("jump_requested",    False),
            "jump_target":       state.get("jump_target",       0.0),
            "saved_at":          time.time(),
        }

        self._write(key, payload)

        logger.debug(
            "save_queue_state — room=%s video=%s processed=%.1fs",
            room_id, video_id, payload["processed_until_s"],
        )

    def load_queue_state(
        self,
        room_id:  str,
        video_id: str,
    ) -> Optional[Dict]:
        """
        Restore queue state for a room/video.

        Returns None if no state found.
        """

        key = f"queue_state:{room_id}:{video_id}"
        data = self._read(key)

        if data:
            logger.info(
                "load_queue_state — room=%s video=%s "
                "processed=%.1fs  RESTORED ✅",
                room_id, video_id,
                data.get("processed_until_s", 0.0),
            )

        return data

    # =========================================================================
    # PROCESSED VIDEOS
    # =========================================================================

    def save_processed_video(
        self,
        room_id:  str,
        video_id: str,
    ) -> None:
        """Mark a video as fully processed for a room."""

        key = f"processed_video:{room_id}"
        self._write(key, {"video_id": video_id, "saved_at": time.time()})

        logger.info(
            "save_processed_video — room=%s video=%s",
            room_id, video_id,
        )

    def load_processed_video(self, room_id: str) -> Optional[str]:
        """Return the fully-processed video_id for a room, or None."""

        key = f"processed_video:{room_id}"
        data = self._read(key)

        if data:
            video_id = data.get("video_id", "")
            logger.info(
                "load_processed_video — room=%s video=%s RESTORED ✅",
                room_id, video_id,
            )
            return video_id

        return None

    # =========================================================================
    # REALTIME TRANSCRIPTS
    # =========================================================================

    def save_realtime_transcripts(
        self,
        room_id:     str,
        video_id:    str,
        transcripts: List[Dict],
    ) -> None:
        """
        Save realtime transcript list to Redis.

        Only saves last 100 entries to keep payload small.
        """

        key = f"realtime:{room_id}:{video_id}"
        payload = {
            "transcripts": transcripts[-100:],
            "saved_at":    time.time(),
        }
        self._write(key, payload)

        logger.debug(
            "save_realtime_transcripts — room=%s video=%s count=%d",
            room_id, video_id, len(transcripts),
        )

    def load_realtime_transcripts(
        self,
        room_id:  str,
        video_id: str,
    ) -> List[Dict]:
        """Restore realtime transcripts. Returns [] if not found."""

        key = f"realtime:{room_id}:{video_id}"
        data = self._read(key)

        if data:
            transcripts = data.get("transcripts", [])
            logger.info(
                "load_realtime_transcripts — room=%s video=%s "
                "count=%d RESTORED ✅",
                room_id, video_id, len(transcripts),
            )
            return transcripts

        return []

    # =========================================================================
    # TRANSCRIPT READY BOUNDARY
    # =========================================================================

    def save_transcript_ready(
        self,
        room_id:  str,
        video_id: str,
        ready_at: float,
    ) -> None:
        """Save how far transcripts are ready."""

        key = f"transcript_ready:{room_id}:{video_id}"
        self._write(key, {"ready_at": ready_at, "saved_at": time.time()})

    def load_transcript_ready(
        self,
        room_id:  str,
        video_id: str,
    ) -> float:
        """Restore transcript ready boundary. Returns 0.0 if not found."""

        key = f"transcript_ready:{room_id}:{video_id}"
        data = self._read(key)

        if data:
            ready = data.get("ready_at", 0.0)
            logger.info(
                "load_transcript_ready — room=%s video=%s "
                "ready=%.1fs RESTORED ✅",
                room_id, video_id, ready,
            )
            return ready

        return 0.0

    # =========================================================================
    # VIDEO METADATA CACHE
    # =========================================================================

    def save_metadata(
        self,
        platform: str,
        video_id: str,
        metadata: Dict,
    ) -> None:
        """Cache video metadata in Redis."""

        key = f"metadata:{platform}:{video_id}"
        payload = {**metadata, "saved_at": time.time()}
        self._write(key, payload)

        logger.debug(
            "save_metadata — platform=%s video=%s title=%s",
            platform, video_id, metadata.get("title", ""),
        )

    def load_metadata(
        self,
        platform: str,
        video_id: str,
    ) -> Optional[Dict]:
        """Restore metadata from Redis cache."""

        key = f"metadata:{platform}:{video_id}"
        data = self._read(key)

        if data:
            logger.debug(
                "load_metadata — platform=%s video=%s RESTORED ✅",
                platform, video_id,
            )
            # Remove internal field
            data.pop("saved_at", None)
            return data

        return None

    # =========================================================================
    # ROOM REGISTRY
    # =========================================================================

    def register_room(self, room_id: str) -> None:
        """
        Register a room with its creation timestamp.
        Used by cleanup job to find rooms older than 24h.
        """

        key = f"room_registry:{room_id}"
        existing = self._read(key)

        # Don't overwrite existing registration
        if not existing:
            self._write(key, {
                "room_id":    room_id,
                "created_at": time.time(),
            })
            logger.info("register_room — room=%s", room_id)

    def get_all_rooms(self) -> List[Dict]:
        """
        Return all registered rooms.
        Used by cleanup job.
        """

        if not self._enabled:
            return list(self._local_cache.get("rooms", {}).values())

        try:
            keys = self._redis.keys("room_registry:*")
            rooms = []

            for key in keys:
                data = self._read(key)
                if data:
                    rooms.append(data)

            return rooms

        except Exception as e:
            logger.warning("get_all_rooms failed: %s", e)
            return []

    def get_room_videos(self, room_id: str) -> List[str]:
        """
        Return all video_ids processed for a room.
        Used by cleanup job.
        """

        if not self._enabled:
            return []

        try:
            keys = self._redis.keys(f"queue_state:{room_id}:*")
            video_ids = []

            for key in keys:
                # key format: queue_state:{room_id}:{video_id}
                parts = key.split(":")
                if len(parts) >= 3:
                    video_ids.append(parts[2])

            return video_ids

        except Exception as e:
            logger.warning("get_room_videos failed: %s", e)
            return []

    # =========================================================================
    # CLEANUP — DELETE ALL ROOM DATA
    # =========================================================================

    def delete_room_state(self, room_id: str) -> None:
        """
        Delete ALL Redis state for a room.

        Called by cleanup job after 24h.
        Deletes:
        - queue_state:{room_id}:*
        - processed_video:{room_id}
        - realtime:{room_id}:*
        - transcript_ready:{room_id}:*
        - room_registry:{room_id}
        """

        if not self._enabled:
            # Clear local cache
            for key in list(self._local_cache.keys()):
                if room_id in key:
                    del self._local_cache[key]
            return

        try:
            patterns = [
                f"queue_state:{room_id}:*",
                f"realtime:{room_id}:*",
                f"transcript_ready:{room_id}:*",
            ]

            exact_keys = [
                f"processed_video:{room_id}",
                f"room_registry:{room_id}",
            ]

            deleted = 0

            for pattern in patterns:
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
                    deleted += len(keys)

            for key in exact_keys:
                self._redis.delete(key)
                deleted += 1

            logger.info(
                "delete_room_state — room=%s  deleted %d keys",
                room_id, deleted,
            )

        except Exception as e:
            logger.warning("delete_room_state failed: %s", e)

    async def restore_room_state(
        self,
        room_id:  str,
        video_id: str,
    ) -> dict:
        """
        Restore ALL state for a room/video from Redis.
        
        Called when:
        - /playback-update gets unknown room
        - /ask detects processed_until = 0 but Redis has data
        - Any endpoint detects missing state
        
        Returns what was restored.
        """

        result = {
            "queue_restored":      False,
            "realtime_restored":   False,
            "processed_until":     0.0,
            "transcript_ready":    0.0,
        }

        # ── Restore queue state ────────────────────────────────────────────────
        from .processing_queue import processing_queue

        queue_saved = self.load_queue_state(room_id, video_id)

        if queue_saved:
            key = (room_id, video_id)
            processing_queue._state[key] = {
                "video_id":          video_id,
                "total_duration_s":  queue_saved.get("total_duration_s",  0.0),
                "current_time":      queue_saved.get("current_time",      0.0),
                "processed_until_s": queue_saved.get("processed_until_s", 0.0),
                "queue":             [],
                "processing":        False,
                "jump_requested":    False,
                "jump_target":       0.0,
            }
            processing_queue._refill_queue(key)

            result["queue_restored"]  = True
            result["processed_until"] = queue_saved.get("processed_until_s", 0.0)

            print(
                f"✅ Queue restored — room={room_id} "
                f"processed={result['processed_until']:.1f}s"
            )

        # ── Restore realtime memory ────────────────────────────────────────────
        from .realtime_memory import realtime_memory

        rt_restored = realtime_memory.restore_from_redis(
            room_id=room_id,
            video_id=video_id,
        )

        if rt_restored:
            result["realtime_restored"] = True
            result["transcript_ready"]  = realtime_memory.get_transcript_ready(
                room_id=room_id,
                video_id=video_id,
            )
            print(
                f"✅ Realtime restored — room={room_id} "
                f"transcript_ready={result['transcript_ready']:.1f}s"
            )

        # ── Register room ──────────────────────────────────────────────────────
        if result["queue_restored"] or result["realtime_restored"]:
            self.register_room(room_id)
            print(
                f"✅ Full restore complete — room={room_id} "
                f"video={video_id}"
            )
        else:
            print(
                f"⚠️ Nothing to restore — room={room_id} "
                f"video={video_id} (fresh start)"
            )

        return result

    # =========================================================================
    # PRIVATE — READ / WRITE
    # =========================================================================

    def _write(self, key: str, payload: Dict) -> None:
        """Write to Redis with 24h TTL, fallback to local cache."""

        if self._enabled:
            try:
                self._redis.set(key, json.dumps(payload), ex=TTL_SECONDS)
                return
            except Exception as e:
                logger.warning("Redis write failed: %s", e)

        # Fallback
        self._local_cache[key] = payload

    def _read(self, key: str) -> Optional[Dict]:
        """Read from Redis, fallback to local cache."""

        if self._enabled:
            try:
                data = self._redis.get(key)
                return json.loads(data) if data else None
            except Exception as e:
                logger.warning("Redis read failed: %s", e)

        return self._local_cache.get(key)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

state_persistence = StatePersistence()