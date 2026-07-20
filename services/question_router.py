"""
Question Router
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class QuestionRouter:

    def __init__(self):
        logger.info("✅ QuestionRouter initialized")

    async def classify(
        self,
        question:      str,
        video_context: Dict,
    ) -> Dict:

        if not question:
            return {"intent": "scene", "confidence": 0.0}

        q = question.lower().strip()

        logger.info("classify — question: %s", question[:80])

        # ── 1: Identity ───────────────────────────────────────────────────
        if self._is_identity_question(q):
            return {"intent": "identity", "confidence": 0.99}

        # ── 2: Conversational ─────────────────────────────────────────────
        if self._is_conversational(q):
            return {"intent": "conversational", "confidence": 0.99}

        # ── 3: Video status ───────────────────────────────────────────────
        if self._is_video_status_question(q):
            return {"intent": "video_status", "confidence": 0.99}

        # ── 4: Processing status ──────────────────────────────────────────
        if self._is_processing_status_question(q):
            return {"intent": "processing_status", "confidence": 0.99}

        # ── 5: Metadata ───────────────────────────────────────────────────
        if self._is_metadata_question(q):
            return {"intent": "metadata", "confidence": 0.95}

        # ── 6: Default — Gemini handles everything else ───────────────────
        # This includes: scene, summary, dialogue, translation,
        # timeline, recommendation, subtitle, and ALL unknowns
        return {"intent": "scene", "confidence": 0.8}

    # =========================================================================
    # DETECTORS
    # =========================================================================

    def _is_identity_question(self, q: str) -> bool:
        keywords = [
            "who are you", "what are you",
            "which ai", "what ai", "which model", "what model",
            "who made you", "who created you", "who built you",
            "are you gpt", "are you chatgpt", "are you gemini",
            "are you claude", "are you an ai", "are you a bot",
            "what is your name", "your name",
            "introduce yourself", "tell me about yourself",
            "what can you do", "how do you work",
        ]
        return any(kw in q for kw in keywords)

    def _is_conversational(self, q: str) -> bool:
        exact = {
            "hi", "hello", "hey", "hii", "hiii",
            "ok", "okay", "k", "kk",
            "lol", "haha", "hehe", "lmao",
            "thanks", "thank you", "thx", "ty",
            "bye", "goodbye", "cya",
            "good", "great", "nice", "cool", "awesome",
            "yes", "no", "yep", "nope", "yeah", "nah",
            "wow", "omg", "oh", "ah", "hmm",
            "bro", "dude", "man",
            "sup", "wassup",
        }

        if q.strip("?!. ") in exact:
            return True

        phrases = [
            "how are you", "how r you", "how're you", "how are u",
            "good morning", "good evening", "good night", "good afternoon",
            "nice to meet", "thank you so much", "thanks a lot",
            "you're welcome", "no problem", "that's great", "that's cool",
            "you're amazing", "you're helpful", "well done", "good job",
        ]

        return any(p in q for p in phrases)

    def _is_video_status_question(self, q: str) -> bool:
        keywords = [
            "video started", "movie started",
            "is it started", "has it started",
            "is it playing", "is the video",
            "is the movie", "started or not",
            "playing or not", "is anything playing",
            "did it start", "has the video",
            "has the movie", "is it on",
        ]
        return any(kw in q for kw in keywords)

    def _is_processing_status_question(self, q: str) -> bool:
        keywords = [
            "how much processed", "how much you processed",
            "you processed how much", "how far processed",
            "processing status", "how far ahead",
            "how much have you", "what have you processed",
            "how much do you know", "have you processed",
            "processing progress", "how much whispered",
            "how far you processed", "what you processed",
            "itne time", "bas itna",
        ]
        return any(kw in q for kw in keywords)

    def _is_metadata_question(self, q: str) -> bool:
        keywords = [
            # What are we watching
            "what are we watching", "what is this movie",
            "what is this video", "what's the movie",
            "which movie", "which film", "which show",
            "what film", "what show", "what movie",
            "what are we", "what is this",
            # Info requests
            "tell me about this", "movie info", "film info",
            "video info", "about this movie", "about this video",
            # Duration
            "how long", "duration", "runtime",
            "how many minutes", "how many hours", "length of",
            # Channel
            "who uploaded", "who made", "which channel",
            "channel name", "who posted", "uploader",
            # Date
            "when was", "release date", "when released",
            "when published", "when did this",
            # Cast
            "who stars", "starring", "cast", "actors",
            "who is in", "who are in",
            # Title
            "title", "movie name", "film name", "video name",
            "what's playing", "what is playing",
        ]
        return any(kw in q for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

question_router = QuestionRouter()