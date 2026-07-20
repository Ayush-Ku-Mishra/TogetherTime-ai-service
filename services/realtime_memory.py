"""
Realtime Memory

Stores raw Whisper transcripts immediately after transcription.
No Gemini, no embedding, no Qdrant — just raw transcript.

This is the FAST memory layer.
Transcripts are available within 1-2 seconds of Whisper finishing.

Shape
-----
{
    room_id: {
        video_id: [
            {
                start:      float,
                end:        float,
                transcript: str,
                language:   str,
                stored_at:  float,   # time.time()
            },
            ...
        ]
    }
}

Does NOT
--------
❌ Call Gemini
❌ Call Qdrant
❌ Call Redis
❌ Summarize anything

Only stores and retrieves raw transcripts.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional
from .state_persistence import state_persistence

logger = logging.getLogger(__name__)

# Max transcripts per room/video in memory
MAX_TRANSCRIPTS = 500


class RealtimeMemory:

    def __init__(self):
        # { room_id: { video_id: [ transcript_dict ] } }
        self._store: Dict[str, Dict[str, List[Dict]]] = {}

        # Track transcript_ready boundary per room/video
        # Shape: { (room_id, video_id): float }
        self._transcript_ready: Dict[tuple, float] = {}

        logger.info("✅ RealtimeMemory initialized")

    # =========================================================================
    # PUBLIC — STORE
    # =========================================================================

    def store_transcript(
        self,
        room_id:    str,
        video_id:   str,
        start:      float,
        end:        float,
        transcript: str,
        language:   str = "",
    ) -> None:

        if not transcript.strip():
            return

        room = self._store.setdefault(room_id, {})
        buf  = room.setdefault(video_id, [])

        buf.append({
            "start":      start,
            "end":        end,
            "transcript": transcript.strip(),
            "language":   language,
            "stored_at":  time.time(),
        })

        if len(buf) > MAX_TRANSCRIPTS:
            room[video_id] = buf[-MAX_TRANSCRIPTS:]

        key = (room_id, video_id)
        current = self._transcript_ready.get(key, 0.0)
        if end > current:
            self._transcript_ready[key] = end

        # ── Persist to Redis ───────────────────────────────────────────────────
        state_persistence.save_realtime_transcripts(
            room_id=room_id,
            video_id=video_id,
            transcripts=room[video_id],
        )
        state_persistence.save_transcript_ready(
            room_id=room_id,
            video_id=video_id,
            ready_at=self._transcript_ready[key],
        )

        logger.info(
            "store_transcript — room=%s  %.1f→%.1fs  "
            "transcript_ready=%.1fs",
            room_id, start, end,
            self._transcript_ready[key],
        )

    def restore_from_redis(
        self,
        room_id:  str,
        video_id: str,
    ) -> bool:
        """
        Restore transcripts from Redis after a restart.

        Called by pipeline_orchestrator before starting a video.
        Returns True if data was restored, False if nothing found.
        """

        transcripts = state_persistence.load_realtime_transcripts(
            room_id=room_id,
            video_id=video_id,
        )

        ready_at = state_persistence.load_transcript_ready(
            room_id=room_id,
            video_id=video_id,
        )

        if not transcripts and ready_at == 0.0:
            return False

        # Restore in-memory store
        room = self._store.setdefault(room_id, {})
        room[video_id] = transcripts

        # Restore boundary
        key = (room_id, video_id)
        self._transcript_ready[key] = ready_at

        logger.info(
            "restore_from_redis — room=%s video=%s "
            "count=%d ready=%.1fs ✅",
            room_id, video_id,
            len(transcripts), ready_at,
        )

        return True

    # =========================================================================
    # PUBLIC — GET TRANSCRIPT READY BOUNDARY
    # =========================================================================

    def get_transcript_ready(
        self,
        room_id:  str,
        video_id: str,
    ) -> float:
        """
        Return how far transcripts are available in seconds.

        This replaces processed_until for the purpose of
        deciding if the AI can answer a question.

        Transcript available = AI can answer.
        Gemini finished = nice to have, not required.
        """
        return self._transcript_ready.get((room_id, video_id), 0.0)

    # =========================================================================
    # PUBLIC — GET TRANSCRIPT AT TIMESTAMP
    # =========================================================================

    def get_transcript_at(
        self,
        room_id:   str,
        video_id:  str,
        timestamp: float,
        window:    float = 60.0,
    ) -> Optional[str]:
        """
        Return combined transcript text near a given timestamp.

        Searches within [timestamp - window, timestamp + window].
        Returns None if no transcript found.
        """

        transcripts = self._get_transcripts_in_range(
            room_id=room_id,
            video_id=video_id,
            start=max(0, timestamp - window),
            end=timestamp + window,
        )

        if not transcripts:
            return None

        combined = " ".join(t["transcript"] for t in transcripts)
        return combined.strip() or None

    # =========================================================================
    # PUBLIC — GET RECENT TRANSCRIPTS
    # =========================================================================

    def get_recent_transcripts(
        self,
        room_id:   str,
        video_id:  str,
        before:    float,
        limit:     int = 5,
    ) -> List[Dict]:
        """
        Return the last N transcript chunks before a timestamp.

        Used to give the AI recent dialogue context.
        """

        all_t = self._store.get(room_id, {}).get(video_id, [])

        filtered = [
            t for t in all_t
            if t.get("end", 0) <= before
        ]

        filtered.sort(key=lambda t: t.get("start", 0))

        return filtered[-limit:] if len(filtered) > limit else filtered

    # =========================================================================
    # PUBLIC — IS READY FOR QUESTION
    # =========================================================================

    def is_ready_for_question(
        self,
        room_id:          str,
        video_id:         str,
        current_timestamp: float,
        buffer_seconds:   float = 5.0,
    ) -> bool:
        """
        Return True if we have transcripts covering current_timestamp.

        Used by the processing guard in /ask — checks transcript
        availability, NOT Gemini/Qdrant completion.
        """
        ready = self.get_transcript_ready(room_id, video_id)
        return ready >= current_timestamp - buffer_seconds

    # =========================================================================
    # PUBLIC — CLEAR
    # =========================================================================

    def clear(self, room_id: str) -> None:
        """Clear all transcripts for a room."""
        self._store.pop(room_id, None)

        keys = [k for k in self._transcript_ready if k[0] == room_id]
        for k in keys:
            del self._transcript_ready[k]

        logger.info("RealtimeMemory cleared — room=%s", room_id)

    # =========================================================================
    # PRIVATE
    # =========================================================================

    def _get_transcripts_in_range(
        self,
        room_id: str,
        video_id: str,
        start:   float,
        end:     float,
    ) -> List[Dict]:

        all_t = self._store.get(room_id, {}).get(video_id, [])

        return [
            t for t in all_t
            if not (t.get("end", 0) < start or t.get("start", 0) > end)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

realtime_memory = RealtimeMemory()