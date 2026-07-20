"""
Timeline Service

Answers questions about time without calling the LLM.

Responsibilities
----------------
✅ Parse timestamps from natural language
✅ Convert seconds ↔ HH:MM:SS
✅ Detect relative time ("30 seconds ago")
✅ Detect absolute time ("at 1:15")
✅ Detect special positions ("ending", "beginning")
✅ Calculate remaining duration
✅ Calculate progress percentage

Does NOT
--------
❌ Call Gemini
❌ Call Whisper
❌ Query Redis
❌ Query Qdrant
❌ Build prompts
❌ Store anything

Only performs time calculations and timestamp parsing.

Examples
--------
User: "What happened at 1:15?"
→ parse_timestamp_from_question() → 75.0

User: "What happened 30 seconds ago?"
→ parse_timestamp_from_question() → (current - 30)

User: "Tell me about the ending"
→ parse_timestamp_from_question() → (duration - 30)

User: "Where am I?"
→ seconds_to_hhmmss(142.6) → "2:22"
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TimelineService
# ─────────────────────────────────────────────────────────────────────────────

class TimelineService:
    """
    Pure time logic — no AI, no database, no external dependencies.

    All methods are synchronous (no async) because they only do
    math and string parsing.
    """

    def __init__(self):
        logger.info("✅ TimelineService initialized")

    # =========================================================================
    # PUBLIC — TIME CONVERSION
    # =========================================================================

    def seconds_to_hhmmss(self, seconds: float) -> str:
        """
        Convert seconds to human-readable timestamp.

        Examples
        --------
        142.6  →  "2:22"
        3661.5 →  "1:01:01"
        45.0   →  "0:45"
        0.0    →  "0:00"

        Parameters
        ----------
        seconds : total duration in seconds

        Returns
        -------
        String in format "H:MM:SS" or "M:SS" (leading zero hour omitted).
        """

        if seconds < 0:
            seconds = 0

        total_sec = int(seconds)

        hours   = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs    = total_sec % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def hhmmss_to_seconds(self, timestamp: str) -> float:
        """
        Convert HH:MM:SS or MM:SS to seconds.

        Examples
        --------
        "2:22"     →  142.0
        "1:01:01"  →  3661.0
        "0:45"     →  45.0
        "1:15:30"  →  4530.0

        Parameters
        ----------
        timestamp : string in format "H:MM:SS", "M:SS", or "HH:MM:SS"

        Returns
        -------
        Total seconds as float, or 0.0 if parsing fails.
        """

        if not timestamp:
            return 0.0

        parts = timestamp.strip().split(":")

        try:
            if len(parts) == 3:
                # HH:MM:SS
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + int(s)

            elif len(parts) == 2:
                # MM:SS
                m, s = parts
                return int(m) * 60 + int(s)

            elif len(parts) == 1:
                # Just seconds
                return float(parts[0])

            else:
                logger.warning("Invalid timestamp format: %s", timestamp)
                return 0.0

        except ValueError:
            logger.warning("Failed to parse timestamp: %s", timestamp)
            return 0.0

    # =========================================================================
    # PUBLIC — CURRENT POSITION INFO
    # =========================================================================

    def get_current_timestamp(
        self,
        current_playback: float,
    ) -> str:
        """
        Return the current playback position as HH:MM:SS.

        Parameters
        ----------
        current_playback : current position in seconds

        Returns
        -------
        Formatted timestamp string e.g. "2:22"
        """

        return self.seconds_to_hhmmss(current_playback)

    def get_duration(
        self,
        duration_seconds: float,
    ) -> str:
        """
        Return the total video duration as HH:MM:SS.

        Parameters
        ----------
        duration_seconds : total video length in seconds

        Returns
        -------
        Formatted timestamp string e.g. "2:28"
        """

        return self.seconds_to_hhmmss(duration_seconds)

    def get_progress_info(
        self,
        current_playback: float,
        duration_seconds: float,
    ) -> dict:
        """
        Calculate complete progress information.

        Returns
        -------
        {
            "current":         str,    "2:22"
            "duration":        str,    "2:28"
            "currentSeconds":  float,  142.6
            "durationSeconds": float,  148.0
            "remaining":       str,    "0:05"
            "remainingSeconds": float, 5.4
            "progressPercent": float,  96.4
        }
        """

        current_str  = self.seconds_to_hhmmss(current_playback)
        duration_str = self.seconds_to_hhmmss(duration_seconds)

        remaining_sec = max(0, duration_seconds - current_playback)
        remaining_str = self.seconds_to_hhmmss(remaining_sec)

        if duration_seconds > 0:
            progress_pct = (current_playback / duration_seconds) * 100
        else:
            progress_pct = 0.0

        return {
            "current":          current_str,
            "duration":         duration_str,
            "currentSeconds":   current_playback,
            "durationSeconds":  duration_seconds,
            "remaining":        remaining_str,
            "remainingSeconds": remaining_sec,
            "progressPercent":  round(progress_pct, 1),
        }

    # =========================================================================
    # PUBLIC — TIMESTAMP PARSING FROM QUESTIONS
    # =========================================================================

    def parse_timestamp_from_question(
        self,
        question:         str,
        current_playback: float,
        duration_seconds: float,
    ) -> Optional[float]:
        """
        Extract a timestamp from a natural language question.

        Patterns detected
        -----------------
        "at 1:15"          → 75.0
        "at 1:15:30"       → 4530.0
        "30 seconds ago"   → current - 30
        "1 minute ago"     → current - 60
        "the ending"       → duration - 30
        "the beginning"    → 0.0
        "the start"        → 0.0

        Parameters
        ----------
        question         : raw user question
        current_playback : current playback position in seconds
        duration_seconds : total video duration in seconds

        Returns
        -------
        Requested timestamp in seconds, or None if no timestamp detected.
        """

        q = question.lower().strip()

        # ── Pattern 1 : Absolute timestamp "at 1:15" ──────────────────────
        # Matches "at 1:15", "at 1:15:30", "around 2:30", "go to 0:45"
        match = re.search(
            r'(?:at|around|go\s+to|skip\s+to)\s+(\d{1,2}:\d{2}(?::\d{2})?)',
            q
        )

        if match:
            timestamp_str = match.group(1)
            seconds = self.hhmmss_to_seconds(timestamp_str)
            logger.info(
                "parse_timestamp: absolute — '%s' → %.1fs",
                timestamp_str, seconds,
            )
            return seconds

        # ── Pattern 2 : Relative past "30 seconds ago" ────────────────────
        # Matches "30 seconds ago", "1 minute ago", "2 minutes ago"
        match = re.search(r'(\d+)\s+(second|minute)s?\s+ago', q)

        if match:
            value = int(match.group(1))
            unit  = match.group(2)

            if unit == "minute":
                value *= 60

            requested = max(0, current_playback - value)

            logger.info(
                "parse_timestamp: relative past — %d %ss ago → %.1fs",
                value, unit, requested,
            )

            return requested

        # ── Pattern 3 : Special positions ──────────────────────────────────
        # "ending", "end", "beginning", "start"

        if any(word in q for word in ["ending", "end"]):
            # Last 30 seconds
            requested = max(0, duration_seconds - 30)
            logger.info(
                "parse_timestamp: ending → %.1fs",
                requested,
            )
            return requested

        if any(word in q for word in ["beginning", "start", "first"]):
            logger.info("parse_timestamp: beginning → 0.0s")
            return 0.0

        # ── No timestamp detected ──────────────────────────────────────────
        return None

    # =========================================================================
    # PUBLIC — TIMESTAMP VALIDATION
    # =========================================================================

    def is_future_timestamp(
        self,
        requested:   float,
        current:     float,
        buffer_sec:  float = 5.0,
    ) -> bool:
        """
        Check if a requested timestamp is ahead of current playback.

        Used to determine if the AI should answer or say
        "I haven't processed that part yet."

        Parameters
        ----------
        requested  : timestamp user asked about (seconds)
        current    : current playback position (seconds)
        buffer_sec : safety margin (default 5s)

        Returns
        -------
        True if requested > current + buffer, False otherwise.
        """

        return requested > current + buffer_sec

    def is_before_processed(
        self,
        requested:       float,
        processed_until: float,
        buffer_sec:      float = 5.0,
    ) -> bool:
        """
        Check if a requested timestamp has been processed by the pipeline.

        Parameters
        ----------
        requested       : timestamp user asked about (seconds)
        processed_until : how far the Whisper/Gemini pipeline has reached
        buffer_sec      : safety margin (default 5s)

        Returns
        -------
        True if requested <= processed_until - buffer, False otherwise.
        """

        return requested <= processed_until - buffer_sec


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

timeline_service = TimelineService()