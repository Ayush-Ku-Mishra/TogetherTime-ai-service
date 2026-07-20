"""
Scene Memory

The central storage service for the AI backend.

Every scene flows through here — nothing else touches Redis or Qdrant directly.

Flow
----

Chunk
  │
  ▼
SceneMemory.process_chunk()
  │
  ▼
Scene Summarizer
  │
  ▼
Scene Object
  │
  ├──────────────────────┐
  ▼                      ▼
Redis                  Qdrant
(Rolling Memory)   (Semantic Memory)

Responsibilities
----------------
✅ Own scene summarization
✅ Own Redis storage
✅ Own Qdrant storage
✅ Health checks for all three
✅ process_chunks() for orchestrator
✅ store() for direct scene injection

Does NOT
---------
❌ Download audio
❌ Run Whisper
❌ Build prompts
❌ Call Gemini directly
❌ Answer questions
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .scene_summarizer import scene_summarizer
from .qdrant_service   import qdrant_service
from memory.rolling_memory import RollingMemory

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scene Memory
# ─────────────────────────────────────────────────────────────────────────────

class SceneMemory:
    """
    Central owner of scene storage.

    Two ways to add a scene
    -----------------------
    1. process_chunks()   — give raw chunks, SceneMemory summarizes + stores
    2. store()            — give a ready-made scene dict, SceneMemory stores it

    Storage targets
    ---------------
    - Redis  (RollingMemory)  — fast recent lookup
    - Qdrant (QdrantService)  — semantic vector search
    """

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self):
        self._redis = RollingMemory()
        logger.info("✅ Scene Memory initialized")

    # =========================================================================
    # PUBLIC — PROCESS CHUNKS
    # =========================================================================

    async def process_chunks(
        self,
        *,
        room_id:  str,
        video_id: str,
        chunks:   List[Dict],
        source:   str = "whisper",
    ) -> Dict:
        """
        Summarize each chunk and store the resulting scene.

        Called by PipelineOrchestrator — it hands chunks here
        and SceneMemory decides everything else.

        Parameters
        ----------
        room_id  : watch party room
        video_id : YouTube / platform video id
        chunks   : output of SubtitleChunker.build_chunks()
        source   : "whisper" | "frontend"

        Returns
        -------
        {
            stored:        int   — number of scenes stored
            failed:        int   — chunks that produced no scene
            summary:       str   — last scene summary (for caller logging)
            processing_time: float
        }
        """

        t0 = time.perf_counter()

        stored  = 0
        failed  = 0
        last_summary = ""

        logger.info(
            "Processing %d chunk(s) — room=%s video=%s source=%s",
            len(chunks), room_id, video_id, source,
        )

        for index, chunk in enumerate(chunks, start=1):

            logger.info(
                "  Chunk %d/%d  %.1f → %.1fs",
                index, len(chunks),
                chunk.get("start", 0),
                chunk.get("end",   0),
            )

            scene = await self._summarize(chunk)

            if not scene:
                logger.warning("  Chunk %d produced no scene", index)
                failed += 1
                continue

            scene["video_id"] = video_id
            scene["source"]   = source

            await self._store_to_redis(room_id, scene)
            await self._store_to_qdrant(room_id, scene)

            last_summary = scene.get("summary", "")
            stored += 1

            logger.info(
                "  ✅ Scene stored | %.1f-%.1fs | %s",
                scene.get("start", 0),
                scene.get("end",   0),
                last_summary[:60],
            )

        elapsed = round(time.perf_counter() - t0, 2)

        logger.info(
            "process_chunks done — stored=%d failed=%d time=%.2fs",
            stored, failed, elapsed,
        )

        return {
            "stored":          stored,
            "failed":          failed,
            "summary":         last_summary,
            "processing_time": elapsed,
        }

    # =========================================================================
    # PUBLIC — STORE (direct injection)
    # =========================================================================

    async def store(
        self,
        room_id: str,
        scene:   Dict,
    ) -> bool:
        """
        Store a ready-made scene dict directly into Redis + Qdrant.

        Use this when the scene has already been summarized externally.

        Parameters
        ----------
        room_id : watch party room
        scene   : {
                    start, end, duration,
                    summary, transcript (optional),
                    keywords (optional),
                    video_id, source, model
                  }

        Returns
        -------
        True if both stores succeeded, False otherwise.
        """

        if not scene:
            logger.warning("store() called with empty scene")
            return False

        redis_ok  = await self._store_to_redis(room_id, scene)
        qdrant_ok = await self._store_to_qdrant(room_id, scene)

        success = redis_ok and qdrant_ok

        logger.info(
            "store() — redis=%s qdrant=%s | %.1f-%.1fs",
            redis_ok, qdrant_ok,
            scene.get("start", 0),
            scene.get("end",   0),
        )

        return success

    # =========================================================================
    # PUBLIC — HEALTH CHECKS
    # =========================================================================

    async def summarizer_healthy(self) -> bool:
        """
        Return True if the scene summarizer has an LLM available.
        """
        return scene_summarizer.enabled

    async def redis_healthy(self) -> bool:
        """
        Return True if Redis is reachable.
        """
        try:
            if not self._redis.enabled:
                return False
            self._redis.redis.ping()
            return True
        except Exception as e:
            logger.warning("Redis health check failed: %s", e)
            return False

    async def qdrant_healthy(self) -> bool:
        """
        Return True if Qdrant is connected.
        """
        try:
            if not qdrant_service.enabled:
                return False
            # Light check — just list collections
            qdrant_service.client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", e)
            return False

    # =========================================================================
    # PRIVATE — SUMMARIZE
    # =========================================================================

    async def _summarize(self, chunk: Dict) -> Optional[Dict]:
        """
        Call SceneSummarizer to convert a chunk into a scene dict.

        Returns None if summarization fails or produces empty output.
        """
        try:
            scene = await scene_summarizer.summarize(chunk)
            return scene if scene else None
        except Exception:
            logger.exception("_summarize failed")
            return None

    # =========================================================================
    # PRIVATE — REDIS STORAGE
    # =========================================================================

    async def _store_to_redis(
        self,
        room_id: str,
        scene:   Dict,
    ) -> bool:
        """
        Append scene summary to the rolling Redis history for this room.

        Stores a lightweight summary record — not the full scene —
        to keep Redis payloads small.
        """
        try:
            record = {
                "timestamp": scene.get("start", 0),
                "summary":   scene.get("summary", ""),
                "video_id":  scene.get("video_id", ""),
                "source":    scene.get("source", ""),
            }
            self._redis.add_scene_summary(room_id, record)
            return True
        except Exception:
            logger.exception("_store_to_redis failed")
            return False

    # =========================================================================
    # PRIVATE — QDRANT STORAGE
    # =========================================================================

    async def _store_to_qdrant(
        self,
        room_id: str,
        scene:   Dict,
    ) -> bool:
        """
        Embed and upsert the scene into Qdrant for semantic search.
        """
        try:
            await qdrant_service.store_scene(room_id, scene)
            return True
        except Exception:
            logger.exception("_store_to_qdrant failed")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

scene_memory = SceneMemory()