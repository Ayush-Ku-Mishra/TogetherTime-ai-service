"""
AI Orchestrator
"""

from __future__ import annotations

import logging
from typing import Dict

from .video_context       import video_context as video_context_service
from .context_builder     import context_builder
from .llm_service         import llm_service
from .question_router     import question_router
from .conversation_memory import conversation_memory
from .timeline_service    import timeline_service
from .youtube_service     import parse_iso_duration

logger = logging.getLogger(__name__)


class AIOrchestrator:

    def __init__(self):
        logger.info("✅ AIOrchestrator initialized")

    # =========================================================================
    # PUBLIC
    # =========================================================================

    async def answer(
        self,
        *,
        room_id:           str,
        video_id:          str,
        platform:          str,
        question:          str,
        user_id:           str   = "",
        current_timestamp: float = 0.0,
    ) -> Dict:

        try:
            # ── Step 1: Build context ─────────────────────────────────────
            ctx = await video_context_service.build(
                room_id=room_id,
                video_id=video_id,
                platform=platform,
                question=question,
                current_timestamp=current_timestamp,
            )

            # ── Step 2: Spoiler guard ─────────────────────────────────────
            if not ctx["spoilerAllowed"]:
                logger.info("Spoiler blocked — room=%s", room_id)
                return {
                    "answer":  ctx["spoilerMessage"],
                    "intent":  "spoiler_blocked",
                    "model":   "direct",
                    "success": True,
                }

            # ── Step 3: Conversation history ──────────────────────────────
            ctx["conversationHistory"] = conversation_memory.get_history(
                room_id=room_id,
                limit=10,
            )

            # ── Step 4: Classify intent ───────────────────────────────────
            classification = await question_router.classify(
                question=question,
                video_context=ctx,
            )

            intent     = classification["intent"]
            confidence = classification["confidence"]
            ctx["intent"] = classification

            logger.info("Intent: %s  (%.2f)", intent, confidence)

            # ── Step 5: Detect video state ────────────────────────────────
            video_is_playing = (
                bool(video_id)
                and ctx["timeline"].get("durationSeconds", 0) > 0
            )

            # ── Step 6: Record user message ───────────────────────────────
            conversation_memory.add_user_message(
                room_id=room_id,
                content=question,
            )

            # ── Step 7: Dispatch ──────────────────────────────────────────
            answer, model = await self._dispatch(
                intent=intent,
                question=question,
                ctx=ctx,
                video_is_playing=video_is_playing,
            )

            # ── Step 8: Record AI message ─────────────────────────────────
            conversation_memory.add_ai_message(
                room_id=room_id,
                content=answer,
            )

            return {
                "answer":  answer,
                "intent":  intent,
                "model":   model,
                "success": True,
            }

        except Exception:
            logger.exception("AIOrchestrator.answer failed")
            return {
                "answer":  "Something went wrong. Please try again.",
                "intent":  "error",
                "model":   "none",
                "success": False,
            }

    # =========================================================================
    # PRIVATE — DISPATCHER
    # =========================================================================

    async def _dispatch(
        self,
        intent:           str,
        question:         str,
        ctx:              Dict,
        video_is_playing: bool = False,
    ) -> tuple[str, str]:

        # ── 1. Identity ───────────────────────────────────────────────────────
        if intent == "identity":
            return self._answer_identity(), "direct"

        # ── 2. Conversational ─────────────────────────────────────────────────
        if intent == "conversational":
            return self._answer_conversational(
                question, video_is_playing
            ), "direct"

        # ── 3. Video status ───────────────────────────────────────────────────
        if intent == "video_status":
            return self._answer_video_status(
                ctx, video_is_playing
            ), "direct"

        # ── 4. Processing status ──────────────────────────────────────────────
        if intent == "processing_status":
            return self._answer_processing_status(ctx), "direct"

        # ── 5. Metadata ───────────────────────────────────────────────────────
        if intent == "metadata":
            if video_is_playing:
                return self._answer_metadata(question, ctx)
            return await self._answer_with_llm(
                question, ctx, video_is_playing
            )

        # ── 6. EVERYTHING ELSE → GEMINI ALWAYS ───────────────────────────────
        # No blocking. No hardcoding.
        # Just tell Gemini the current state and let it decide
        # what to answer and what to say it doesn't know yet.
        return await self._answer_with_llm(
            question, ctx, video_is_playing
        )

    # =========================================================================
    # PRIVATE — HARDCODED HANDLERS (only 5)
    # =========================================================================

    def _answer_identity(self) -> str:
        return (
            "I'm **TogetherTime AI** 🤖 — your real-time watch party companion!\n\n"
            "I'm powered by **Google Gemini** and built to help you:\n"
            "🎬 Understand scenes and characters\n"
            "💬 Get exact dialogue and translations\n"
            "📖 Summarize what's happened so far\n"
            "🎯 Recommend similar movies\n"
            "🛡️ Stay spoiler-free\n\n"
            "Start a video and I'll be right with you! 🍿"
        )

    def _answer_conversational(
        self,
        question:         str,
        video_is_playing: bool,
    ) -> str:

        q = question.lower().strip().strip("?!. ")

        if q in ["hi", "hello", "hey", "hii", "hiii",
                 "sup", "wassup", "what's up", "whats up"]:
            if video_is_playing:
                return (
                    "Hey! 👋 I'm here and watching with you! "
                    "Ask me anything about the scene, characters or dialogue 🎬"
                )
            return (
                "Hey! 👋 I'm TogetherTime AI, your watch party companion!\n"
                "Start a video and I'll help you understand everything "
                "in real time! 🎬🍿"
            )

        if any(p in question.lower() for p in [
            "how are you", "how r you", "how're you", "how are u"
        ]):
            if video_is_playing:
                return "All good and ready to help! 😄 What do you want to know? 🎬"
            return "Doing great! 😄 Start a video and let's get going! 🎬"

        if any(p in question.lower() for p in [
            "thanks", "thank you", "thx", "ty", "thank"
        ]):
            return "Happy to help! 😊 Ask me anything else! 🎬"

        if any(p in question.lower() for p in [
            "bye", "goodbye", "cya", "see you", "good night"
        ]):
            return "Bye! 👋 Come back for your next watch party! 🎬🍿"

        if any(p in question.lower() for p in [
            "cool", "awesome", "great", "nice", "wow", "amazing"
        ]):
            if video_is_playing:
                return "Glad you like it! 😄 Keep watching and ask me anything! 🎬"
            return "Thanks! 😄 Start a video and let's watch together! 🎬"

        if any(p in question.lower() for p in ["lol", "haha", "hehe", "lmao"]):
            return "😄 Haha! I'm here whenever you need me! 🎬"

        if q in ["ok", "okay", "k", "kk", "yes", "yeah", "yep"]:
            if video_is_playing:
                return "Got it! 👍 Ask me anything about the video! 🎬"
            return "Got it! 👍 Start a video when you're ready! 🎬"

        if video_is_playing:
            return "I'm here! 😊 Ask me anything about the scene! 🎬"
        return "I'm here! 😊 Start a video and I'll help you! 🎬"

    def _answer_video_status(
        self,
        ctx:              Dict,
        video_is_playing: bool,
    ) -> str:

        if not video_is_playing:
            return (
                "❌ No video is playing yet!\n\n"
                "Start a video and I'll be ready to help! 🎬"
            )

        timeline = ctx.get("timeline", {})
        current  = timeline.get("current",  "0:00")
        duration = timeline.get("duration", "0:00")
        progress = timeline.get("progressPercent", 0)
        title    = ctx.get("metadata", {}).get("title", "the video")

        return (
            f"✅ Yes! **{title}** is playing!\n\n"
            f"📍 You are at **{current}** / {duration} "
            f"({progress}% complete)\n\n"
            f"Ask me anything about the scene! 🎬"
        )

    def _answer_processing_status(self, ctx: Dict) -> str:
        try:
            from .realtime_memory   import realtime_memory
            from .state_persistence import state_persistence

            room_id  = ctx["roomId"]
            video_id = ctx["videoId"]
            timeline = ctx.get("timeline", {})

            # ── Get transcript ready ───────────────────────────────────────────
            transcript_ready = realtime_memory.get_transcript_ready(
                room_id=room_id,
                video_id=video_id,
            )

            # ── If 0 — check Redis saved state ────────────────────────────────
            if transcript_ready == 0.0:
                saved_rt = state_persistence.load_transcript_ready(
                    room_id=room_id,
                    video_id=video_id,
                )
                if saved_rt > 0:
                    transcript_ready = saved_rt

            # ── Get processed_until ───────────────────────────────────────────
            processed_until = ctx.get("processedUntil", 0)

            # ── Use best available value ───────────────────────────────────────
            # processed_until comes from queue (queue state)
            # transcript_ready comes from realtime memory
            # Both should match but take the higher one
            best_ready = max(transcript_ready, processed_until)

            current      = ctx.get("currentPlayback",   0)
            duration_sec = timeline.get("durationSeconds", 0)

            ready_str     = timeline_service.seconds_to_hhmmss(best_ready)
            current_str   = timeline_service.seconds_to_hhmmss(current)
            duration_str  = timeline_service.seconds_to_hhmmss(duration_sec)
            buffer        = best_ready - current

            return (
                f"📊 **Processing Status**\n\n"
                f"🎬 You are at: **{current_str}** / {duration_str}\n"
                f"⚡ Transcripts ready until: **{ready_str}**\n"
                f"🧠 Summaries ready until: **{ready_str}**\n"
                f"📦 Buffer ahead: **{buffer:.0f}s**"
            )

        except Exception:
            return "📊 Processing status unavailable right now."

    def _answer_metadata(self, question: str, ctx: Dict) -> tuple[str, str]:
        """
        Question-aware metadata — only answer what was asked.
        """

        metadata = ctx.get("metadata", {})
        timeline = ctx.get("timeline", {})
        q        = question.lower()

        title       = metadata.get("title",       "Unknown")
        channel     = metadata.get("channel",     "Unknown")
        description = metadata.get("description", "")
        published   = metadata.get("publishedAt", "Unknown")
        duration    = timeline.get("duration",    "Unknown")

        # ── Duration only ─────────────────────────────────────────────────
        if any(kw in q for kw in [
            "how long", "duration", "runtime", "length",
            "how many minutes", "how many hours",
        ]):
            return f"⏱️ This is **{duration}** long.", "direct"

        # ── Channel only ──────────────────────────────────────────────────
        if any(kw in q for kw in [
            "who uploaded", "who made", "which channel",
            "channel name", "who posted", "uploader",
        ]):
            return f"📺 Uploaded by **{channel}**", "direct"

        # ── Published date only ───────────────────────────────────────────
        if any(kw in q for kw in [
            "when was", "release date", "when released",
            "when published", "published", "release",
        ]):
            date = published[:10] if len(published) >= 10 else published
            return f"📅 Published on **{date}**", "direct"

        # ── Full metadata (what are we watching, general info) ────────────
        lines = [f"🎬 **{title}**"]

        if channel and channel != "Unknown":
            lines.append(f"📺 Channel: {channel}")

        if duration and duration != "Unknown":
            lines.append(f"⏱️ Duration: {duration}")

        if published and published != "Unknown":
            date = published[:10] if len(published) >= 10 else published
            lines.append(f"📅 Published: {date}")

        if description:
            clean = self._clean_description(description)
            if clean:
                lines.append(f"\n📝 {clean}")

        return "\n".join(lines), "direct"

    def _clean_description(self, description: str) -> str:
        """
        Strip URLs, promo text, hashtags, short filler lines.
        """

        if not description:
            return ""

        import re

        lines       = description.split("\n")
        clean_lines = []

        # Promo phrases to skip entirely
        skip_phrases = [
            "watch on", "subscribe", "follow us", "follow me",
            "click here", "link in", "check out", "visit",
            "like and", "share and", "comment below",
            "hit the bell", "notification", "instagram",
            "twitter", "facebook", "youtube", "telegram",
        ]

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            # Skip lines with URLs
            if "http" in stripped.lower() or "www." in stripped.lower():
                continue

            # Skip promo lines
            if any(phrase in stripped.lower() for phrase in skip_phrases):
                continue

            # Skip lines that are mostly hashtags
            words         = stripped.split()
            hashtag_count = sum(1 for w in words if w.startswith("#"))
            if len(words) > 0 and hashtag_count >= len(words) * 0.5:
                continue

            # Skip very short lines
            if len(stripped) < 20:
                continue

            clean_lines.append(stripped)

        result = " ".join(clean_lines[:3])

        if len(result) > 250:
            result = result[:247] + "..."

        return result

    # =========================================================================
    # PRIVATE — GEMINI HANDLER
    # =========================================================================

    async def _answer_with_llm(
        self,
        question:         str,
        ctx:              Dict,
        video_is_playing: bool = False,
    ) -> tuple[str, str]:

        logger.info("_answer_with_llm — building prompt")

        context = await context_builder.build_context(
            video_context=ctx,
            question=question,
            video_is_playing=video_is_playing,
        )

        answer, model = await llm_service.generate(context["prompt"])

        logger.info("_answer_with_llm — model=%s", model)

        return answer, model


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

ai_orchestrator = AIOrchestrator()