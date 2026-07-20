"""
Spoiler Guard

Protects users from seeing future content before they reach it.

Decides ALLOW or BLOCK for every AI question.

Rules
-----
1. If question mentions a future timestamp → BLOCK
2. If question asks about "the end/ending" and user hasn't reached it → BLOCK
3. If question is about past/current content → ALLOW
4. If no timestamp is detected → ALLOW (safe default)

Examples
--------
Playback = 2:10 / Duration = 2:28

"What happens at the end?"      → BLOCK  (ending = 2:28, user is at 2:10)
"Who dies?"                     → BLOCK  (open-ended future question)
"What happened 20 seconds ago?" → ALLOW  (2:10 - 20 = 1:50, in the past)
"Summarize so far"              → ALLOW  (past content only)
"What just happened?"           → ALLOW  (current scene)

Does NOT
--------
❌ Call Gemini
❌ Build prompts
❌ Store anything

Only makes ALLOW / BLOCK decisions.
"""

from __future__ import annotations

import logging
import re
from typing import Dict

from .timeline_service import timeline_service

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# How far ahead of processed_until we allow questions (seconds)
FUTURE_BUFFER_S: float = 10.0

# Fraction of video remaining that counts as "ending"
# e.g. 0.15 = last 15% of video
ENDING_THRESHOLD: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# SpoilerGuard
# ─────────────────────────────────────────────────────────────────────────────

class SpoilerGuard:
    """
    Runs before Gemini to block spoiler questions.
    """

    def __init__(self):
        logger.info("✅ SpoilerGuard initialized")

    # =========================================================================
    # PUBLIC — CHECK
    # =========================================================================

    def check(
        self,
        question:         str,
        current_playback: float,
        processed_until:  float,
        duration_seconds: float,
    ) -> Dict:
        """
        Decide whether to ALLOW or BLOCK a question.

        Parameters
        ----------
        question         : raw user question
        current_playback : current playback position in seconds
        processed_until  : how far the pipeline has processed
        duration_seconds : total video duration in seconds

        Returns
        -------
        {
            "allowed":         bool,
            "reason":          str,
            "requested_time":  float | None,
            "message":         str,   human-readable block message (if blocked)
        }
        """

        q = question.lower().strip()

        # ── Rule 1 : Check for absolute future timestamps ──────────────────
        # "What happens at 2:25?" when playback is at 2:10
        requested_time = timeline_service.parse_timestamp_from_question(
            question=question,
            current_playback=current_playback,
            duration_seconds=duration_seconds,
        )

        if requested_time is not None:
            if requested_time > current_playback + FUTURE_BUFFER_S:
                logger.info(
                    "SpoilerGuard BLOCK — future timestamp %.1fs > playback %.1fs",
                    requested_time, current_playback,
                )
                requested_str = timeline_service.seconds_to_hhmmss(requested_time)
                current_str   = timeline_service.seconds_to_hhmmss(current_playback)
                return {
                    "allowed":        False,
                    "reason":         "future_timestamp",
                    "requested_time": requested_time,
                    "message": (
                        f"⚠️ You haven't reached {requested_str} yet "
                        f"(you're at {current_str}). "
                        f"I won't spoil what happens there!"
                    ),
                }

        # ── Rule 2 : Check for "ending" questions ─────────────────────────
        # Block if user hasn't reached the ending threshold yet
        if self._is_ending_question(q):
            if duration_seconds > 0:
                progress = current_playback / duration_seconds
                if progress < (1 - ENDING_THRESHOLD):
                    current_str  = timeline_service.seconds_to_hhmmss(current_playback)
                    duration_str = timeline_service.seconds_to_hhmmss(duration_seconds)
                    logger.info(
                        "SpoilerGuard BLOCK — ending question at %.1f%% progress",
                        progress * 100,
                    )
                    return {
                        "allowed":        False,
                        "reason":         "ending_spoiler",
                        "requested_time": duration_seconds,
                        "message": (
                            f"⚠️ You're at {current_str} / {duration_str}. "
                            f"I won't spoil the ending — "
                            f"watch and find out! 🎬"
                        ),
                    }

        # ── Rule 3 : Check for open-ended future spoiler questions ────────
        if self._is_future_spoiler_question(q):
            logger.info("SpoilerGuard BLOCK — future spoiler question")
            return {
                "allowed":        False,
                "reason":         "spoiler_question",
                "requested_time": None,
                "message": (
                    "⚠️ That question might contain spoilers for content "
                    "you haven't reached yet. "
                    "Keep watching to find out! 🎬"
                ),
            }

        # ── ALLOW ─────────────────────────────────────────────────────────
        logger.info("SpoilerGuard ALLOW — safe question")
        return {
            "allowed":        True,
            "reason":         "safe",
            "requested_time": requested_time,
            "message":        "",
        }

    # =========================================================================
    # PRIVATE — PATTERN DETECTORS
    # =========================================================================

    def _is_ending_question(self, q: str) -> bool:
        """
        Detect questions explicitly about the video ending.

        NOT triggered by:
        - "last scene"   → means recently seen scene
        - "last minute"  → means recently seen content
        """

        # Phrases that clearly mean the video's ending
        ending_phrases = [
            "how does it end",
            "how does the movie end",
            "how does this end",
            "what happens at the end",
            "does he die",
            "does she die",
            "do they die",
            "who dies",
            "who survives",
            "who wins",
            "who loses",
            "final scene",
        ]

        # "ending" and "end" only if clearly about the video ending
        end_context_phrases = [
            "the ending",
            "video ending",
            "movie ending",
            "film ending",
            "show ending",
            "tell me the ending",
            "explain the ending",
            "what is the ending",
        ]

        if any(phrase in q for phrase in ending_phrases):
            return True

        if any(phrase in q for phrase in end_context_phrases):
            return True

        return False

    def _is_future_spoiler_question(self, q: str) -> bool:
        """Detect open-ended spoiler questions about future events."""

        spoiler_phrases = [
            "what will happen",
            "what happens next",
            "what happens later",
            "what happens after",
            "will he",
            "will she",
            "will they",
            "is he going to",
            "is she going to",
            "are they going to",
        ]

        return any(phrase in q for phrase in spoiler_phrases)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

spoiler_guard = SpoilerGuard()