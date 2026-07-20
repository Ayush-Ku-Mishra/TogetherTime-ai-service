"""
Cleanup Service

Handles all cleanup operations for rooms.

Two triggers
------------
1. Manual   — DELETE /memory/{room_id} (existing endpoint)
2. Auto     — Background job runs every hour, 
              deletes rooms older than 24h

What gets deleted
-----------------
Per room:
├── Audio files      cache/youtube_audio/{room_id}/
├── Audio clips      cache/audio_clips/{video_id}_*.wav
├── Redis state      All room_registry, queue_state, etc.
├── Qdrant vectors   All points with roomId == room_id
├── Rolling memory   room:{room_id}:memory
└── In-memory stores caption_store, processing_queue, etc.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# 24 hours
MAX_ROOM_AGE_SECONDS = 86400

AUDIO_CACHE_DIR = Path("cache/youtube_audio")
CLIP_CACHE_DIR  = Path("cache/audio_clips")


class CleanupService:

    def __init__(self):
        logger.info("✅ CleanupService initialized")

    # =========================================================================
    # PUBLIC — FULL ROOM CLEANUP
    # =========================================================================

    async def cleanup_room(self, room_id: str) -> Dict:
        """
        Delete everything for a room.

        Called by:
        - Background job (24h auto-cleanup)
        - Manual DELETE /memory/{room_id}

        Returns
        -------
        Summary of what was deleted.
        """

        logger.info("🗑 Starting full cleanup — room=%s", room_id)

        result = {
            "room_id":        room_id,
            "audio_deleted":  False,
            "clips_deleted":  0,
            "redis_cleared":  False,
            "qdrant_cleared": False,
            "memory_cleared": False,
        }

        # ── Step 1: Get all video_ids for this room ────────────────────────
        from .state_persistence import state_persistence
        video_ids = state_persistence.get_room_videos(room_id)

        logger.info(
            "cleanup_room — room=%s  videos=%s",
            room_id, video_ids,
        )

        # ── Step 2: Delete audio files ─────────────────────────────────────
        result["audio_deleted"] = self._delete_audio_files(room_id)

        # ── Step 3: Delete audio clips for each video ──────────────────────
        for video_id in video_ids:
            clips = self._delete_audio_clips(video_id)
            result["clips_deleted"] += clips

        # ── Step 4: Clear Redis state ──────────────────────────────────────
        result["redis_cleared"] = self._clear_redis(room_id)

        # ── Step 5: Clear Qdrant vectors ───────────────────────────────────
        result["qdrant_cleared"] = await self._clear_qdrant(room_id)

        # ── Step 6: Clear in-memory stores ────────────────────────────────
        result["memory_cleared"] = self._clear_memory(room_id)

        logger.info(
            "✅ Cleanup complete — room=%s  result=%s",
            room_id, result,
        )

        return result

    # =========================================================================
    # PUBLIC — BACKGROUND JOB
    # =========================================================================

    async def run_cleanup_loop(self) -> None:
        """
        Background task that runs every hour.
        Finds rooms older than 24h and cleans them up.

        Started in main.py lifespan on startup.
        """

        logger.info("🔄 Cleanup background job started")

        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_rooms()

            except asyncio.CancelledError:
                logger.info("Cleanup job cancelled")
                break

            except Exception as e:
                logger.warning("Cleanup job error: %s", e)
                await asyncio.sleep(60)  # Retry in 1 min on error

    async def _cleanup_old_rooms(self) -> None:
        """Find and clean rooms older than 24h."""

        from .state_persistence import state_persistence

        now   = time.time()
        rooms = state_persistence.get_all_rooms()

        logger.info(
            "Auto-cleanup check — %d room(s) registered",
            len(rooms),
        )

        cleaned = 0

        for room in rooms:
            room_id    = room.get("room_id", "")
            created_at = room.get("created_at", now)
            age        = now - created_at

            if age >= MAX_ROOM_AGE_SECONDS:
                logger.info(
                    "Auto-cleanup: room=%s age=%.1fh — CLEANING",
                    room_id, age / 3600,
                )
                await self.cleanup_room(room_id)
                cleaned += 1

            else:
                remaining = MAX_ROOM_AGE_SECONDS - age
                logger.info(
                    "Auto-cleanup: room=%s age=%.1fh — "
                    "keeping (%.1fh remaining)",
                    room_id, age / 3600, remaining / 3600,
                )

        if cleaned:
            logger.info("Auto-cleanup: cleaned %d room(s)", cleaned)
        else:
            logger.info("Auto-cleanup: nothing to clean")

    # =========================================================================
    # PRIVATE — DELETE AUDIO FILES
    # =========================================================================

    def _delete_audio_files(self, room_id: str) -> bool:
        """
        Delete the room's audio folder.

        Path: cache/youtube_audio/{room_id}/
        Contains full downloaded YouTube audio files.
        """

        room_audio_dir = AUDIO_CACHE_DIR / room_id

        if not room_audio_dir.exists():
            logger.info(
                "_delete_audio_files: no folder found — room=%s",
                room_id,
            )
            return True

        try:
            shutil.rmtree(room_audio_dir)
            logger.info(
                "_delete_audio_files: deleted %s",
                room_audio_dir,
            )
            return True

        except Exception as e:
            logger.warning(
                "_delete_audio_files failed — room=%s: %s",
                room_id, e,
            )
            return False

    def _delete_audio_clips(self, video_id: str) -> int:
        """
        Delete all trimmed clips for a video.

        Path: cache/audio_clips/{video_id}_*.wav
        These are short 30s clips created by AudioTrimmer.
        """

        if not CLIP_CACHE_DIR.exists():
            return 0

        pattern = f"{video_id}_*.wav"
        clips   = list(CLIP_CACHE_DIR.glob(pattern))
        deleted = 0

        for clip in clips:
            try:
                clip.unlink()
                deleted += 1
                logger.debug("Deleted clip: %s", clip.name)
            except Exception as e:
                logger.warning("Failed to delete clip %s: %s", clip.name, e)

        if deleted:
            logger.info(
                "_delete_audio_clips: deleted %d clip(s) for video=%s",
                deleted, video_id,
            )

        return deleted

    # =========================================================================
    # PRIVATE — CLEAR REDIS
    # =========================================================================

    def _clear_redis(self, room_id: str) -> bool:
        """
        Clear all Redis keys for a room.

        Clears:
        - StatePersistence keys (queue_state, realtime, etc.)
        - RollingMemory key    (room:{room_id}:memory)
        """

        try:
            # StatePersistence cleanup
            from .state_persistence import state_persistence
            state_persistence.delete_room_state(room_id)

            # RollingMemory cleanup
            from memory.rolling_memory import RollingMemory
            rolling = RollingMemory()
            rolling.clear(room_id)

            logger.info("_clear_redis — room=%s ✅", room_id)
            return True

        except Exception as e:
            logger.warning("_clear_redis failed — room=%s: %s", room_id, e)
            return False

    # =========================================================================
    # PRIVATE — CLEAR QDRANT
    # =========================================================================

    async def _clear_qdrant(self, room_id: str) -> bool:
        """Delete all Qdrant vectors for a room."""

        try:
            from .qdrant_service import qdrant_service
            await qdrant_service.delete_room(room_id)
            logger.info("_clear_qdrant — room=%s ✅", room_id)
            return True

        except Exception as e:
            logger.warning("_clear_qdrant failed — room=%s: %s", room_id, e)
            return False

    # =========================================================================
    # PRIVATE — CLEAR IN-MEMORY
    # =========================================================================

    def _clear_memory(self, room_id: str) -> bool:
        """
        Clear all in-memory stores for a room.

        Clears:
        - pipeline_orchestrator  (caption_store, processing_status)
        - processing_queue       (queue state)
        - conversation_memory    (chat history)
        - realtime_memory        (transcripts)
        - video_metadata cache   (title/channel cache)
        """

        try:
            from .pipeline_orchestrator import pipeline_orchestrator
            pipeline_orchestrator.clear_room(room_id)

            from .conversation_memory import conversation_memory
            conversation_memory.clear(room_id)

            from .realtime_memory import realtime_memory
            realtime_memory.clear(room_id)

            logger.info("_clear_memory — room=%s ✅", room_id)
            return True

        except Exception as e:
            logger.warning("_clear_memory failed — room=%s: %s", room_id, e)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

cleanup_service = CleanupService()