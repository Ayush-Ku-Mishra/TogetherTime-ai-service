"""
Subtitle Service

Single source of truth for subtitle/caption data.

The AI never cares whether subtitles came from Whisper, YouTube captions,
or Redis — this service hides that completely.

Sources (priority order)
------------------------
1. In-memory caption store   (frontend captions via /subtitles/push)
2. Qdrant scene transcripts  (Whisper captions stored during processing)

Why two sources?
----------------
When using Whisper pipeline:
  → subtitles go to Qdrant scenes (as "transcript" field)
  → _caption_store is EMPTY

When using frontend captions:
  → subtitles go to _caption_store
  → Qdrant scenes may have no transcript

get_last_subtitles_with_fallback() handles both cases automatically.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .pipeline_orchestrator import get_captions, _caption_store
from .qdrant_service        import qdrant_service

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_S: float = 30.0


class SubtitleService:
    """
    Single entry point for all subtitle retrieval.

    Hides whether subtitles came from Whisper, YouTube captions, or Redis.
    """

    def __init__(self):
        logger.info("✅ SubtitleService initialized")

    # =========================================================================
    # PUBLIC — GET CURRENT SUBTITLES
    # =========================================================================

    def get_current_subtitles(
        self,
        room_id:          str,
        video_id:         str,
        current_playback: float,
        window_seconds:   float = DEFAULT_WINDOW_S,
    ) -> List[Dict]:
        """Return subtitles near the current playback position."""

        subtitles = get_captions(
            room_id=room_id,
            video_id=video_id,
            current_timestamp=current_playback,
            window_seconds=window_seconds,
        )

        logger.info(
            "get_current_subtitles — room=%s  playback=%.1fs  found=%d",
            room_id, current_playback, len(subtitles),
        )

        return subtitles

    # =========================================================================
    # PUBLIC — GET SUBTITLES AT TIMESTAMP
    # =========================================================================

    def get_subtitles_at(
        self,
        room_id:        str,
        video_id:       str,
        timestamp:      float,
        window_seconds: float = DEFAULT_WINDOW_S,
    ) -> List[Dict]:
        """Return subtitles near a specific timestamp."""

        subtitles = get_captions(
            room_id=room_id,
            video_id=video_id,
            current_timestamp=timestamp,
            window_seconds=window_seconds,
        )

        logger.info(
            "get_subtitles_at — room=%s  timestamp=%.1fs  found=%d",
            room_id, timestamp, len(subtitles),
        )

        return subtitles

    # =========================================================================
    # PUBLIC — GET SUBTITLES BETWEEN
    # =========================================================================

    def get_subtitles_between(
        self,
        room_id:  str,
        video_id: str,
        start:    float,
        end:      float,
    ) -> List[Dict]:
        """Return all subtitles within a time range [start, end]."""

        if start > end:
            start, end = end, start

        all_subs = _caption_store.get(room_id, {}).get(video_id, [])

        filtered = [
            s for s in all_subs
            if start <= s.get("timestamp", 0) <= end
        ]

        filtered.sort(key=lambda s: s.get("timestamp", 0))

        logger.info(
            "get_subtitles_between — room=%s  %.1fs-%.1fs  found=%d",
            room_id, start, end, len(filtered),
        )

        return filtered

    # =========================================================================
    # PUBLIC — GET LAST SUBTITLES
    # =========================================================================

    def get_last_subtitles(
        self,
        room_id:  str,
        video_id: str,
        limit:    int = 10,
    ) -> List[Dict]:
        """
        Return the most recent N subtitles from the caption store.

        Only returns frontend captions (_caption_store).
        If you need Whisper captions too, use get_last_subtitles_with_fallback().
        """

        all_subs = _caption_store.get(room_id, {}).get(video_id, [])

        if not all_subs:
            return []

        sorted_subs = sorted(all_subs, key=lambda s: s.get("timestamp", 0))
        recent      = sorted_subs[-limit:]

        logger.info(
            "get_last_subtitles — room=%s  limit=%d  found=%d",
            room_id, limit, len(recent),
        )

        return recent

    # =========================================================================
    # PUBLIC — GET LAST SUBTITLES WITH QDRANT FALLBACK  (Fix 2)
    # =========================================================================

    async def get_last_subtitles_with_fallback(
        self,
        room_id:  str,
        video_id: str,
        limit:    int = 10,
    ) -> List[Dict]:
        """
        Return the most recent N subtitles from any available source.

        Priority
        --------
        1. Caption store (_caption_store)
           → populated when frontend sends captions via /subtitles/push
           → empty when using Whisper-only pipeline

        2. Qdrant scene transcripts  (fallback)
           → populated by Whisper pipeline
           → each scene has a "transcript" field with raw speech text

        This means subtitles will always be available regardless of
        which ingestion path was used.

        Why is this needed?
        -------------------
        Whisper stores transcripts inside Qdrant scenes:
            scene = {
                "start": 0.0,
                "end": 28.5,
                "summary": "...",
                "transcript": "Never gonna give you up...",  ← here
            }

        The caption store (_caption_store) is ONLY filled by:
            POST /subtitles/push  (frontend sends captions)

        So after a Whisper-only process-video call:
            _caption_store = {}       ← empty
            Qdrant scenes  = 5 scenes ← has transcripts

        This method checks caption store first, then Qdrant.
        """

        # ── Source 1: Caption store (frontend captions) ───────────────────
        from_store = self.get_last_subtitles(room_id, video_id, limit)

        if from_store:
            logger.info(
                "get_last_subtitles_with_fallback — "
                "room=%s  source=caption_store  found=%d",
                room_id, len(from_store),
            )
            return from_store

        # ── Source 2: Qdrant scene transcripts (Whisper captions) ─────────
        logger.info(
            "get_last_subtitles_with_fallback — "
            "room=%s  caption_store empty, trying Qdrant",
            room_id,
        )

        try:
            scenes = await qdrant_service.get_video_scenes(
                room_id=room_id,
                video_id=video_id,
            )

            if not scenes:
                logger.info(
                    "get_last_subtitles_with_fallback — "
                    "room=%s  Qdrant also empty",
                    room_id,
                )
                return []

            # Convert each scene's transcript into a subtitle-shaped dict
            subtitles = []

            for scene in scenes:
                transcript = scene.get("transcript", "").strip()

                if not transcript:
                    continue

                subtitles.append({
                    "text":      transcript[:300],
                    "timestamp": scene.get("start", 0),
                    "duration":  (
                        scene.get("end", 0) - scene.get("start", 0)
                    ),
                    "source":    "qdrant_scene",
                })

            # Sort chronologically, return last N
            subtitles.sort(key=lambda s: s.get("timestamp", 0))
            result = subtitles[-limit:]

            logger.info(
                "get_last_subtitles_with_fallback — "
                "room=%s  source=qdrant  found=%d",
                room_id, len(result),
            )

            return result

        except Exception as e:
            logger.warning(
                "get_last_subtitles_with_fallback Qdrant error: %s", e
            )
            return []

    # =========================================================================
    # PUBLIC — SEARCH
    # =========================================================================

    def search(
        self,
        room_id:  str,
        video_id: str,
        keyword:  str,
        limit:    int = 10,
    ) -> List[Dict]:
        """Keyword search across all stored subtitles."""

        if not keyword:
            return []

        keyword_lower = keyword.lower()
        all_subs      = _caption_store.get(room_id, {}).get(video_id, [])

        matches = [
            s for s in all_subs
            if keyword_lower in s.get("text", "").lower()
        ]

        matches.sort(key=lambda s: s.get("timestamp", 0))
        results = matches[:limit]

        logger.info(
            "search — room=%s  keyword='%s'  matches=%d",
            room_id, keyword, len(results),
        )

        return results

    # =========================================================================
    # PUBLIC — FORMAT FOR PROMPT
    # =========================================================================

    def format_for_prompt(
        self,
        subtitles: List[Dict],
        max_chars: int = 1000,
    ) -> str:
        """Convert a list of subtitles into prompt-friendly text."""

        if not subtitles:
            return "No subtitles available."

        from .timeline_service import timeline_service

        lines = []

        for sub in subtitles:
            timestamp = sub.get("timestamp", 0)
            text      = sub.get("text", "").strip()

            if not text:
                continue

            ts_str = timeline_service.seconds_to_hhmmss(timestamp)
            lines.append(f"[{ts_str}] {text}")

        full_text = "\n".join(lines)

        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "..."

        return full_text


subtitle_service = SubtitleService()