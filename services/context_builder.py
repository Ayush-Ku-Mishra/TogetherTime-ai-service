"""
Context Builder
"""

from __future__ import annotations

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class ContextBuilder:

    def __init__(self):
        logger.info("✅ Context Builder initialized")

    # =========================================================================
    # PUBLIC
    # =========================================================================

    async def build_context(
        self,
        *,
        video_context:    Dict,
        question:         str,
        video_is_playing: bool = False,
    ) -> Dict:

        room_id   = video_context["roomId"]
        video_id  = video_context["videoId"]
        metadata  = video_context["metadata"]
        playback  = video_context["currentPlayback"]
        processed = video_context["processedUntil"]
        buffer    = video_context["bufferAhead"]
        timeline  = video_context["timeline"]

        current_scene    = video_context["currentScene"]
        recent_scenes    = video_context["recentScenes"]
        conversation     = video_context["conversationHistory"]
        requested_scene  = video_context.get("requestedScene")
        current_subtitle = video_context.get("currentSubtitle", "")

        realtime_transcript = video_context.get("realtimeTranscript", "")
        recent_realtime     = video_context.get("recentRealtime", [])

        current_text      = self._scene_to_text(current_scene)
        recent_text       = self._scenes_to_text(recent_scenes)
        conversation_text = self._conversation_to_text(conversation)
        requested_text    = self._scene_to_text(requested_scene)

        # ── Sanitize all transcript content before sending to Gemini ──────────
        subtitle_text   = self._sanitize_for_gemini(
            current_subtitle or "No subtitles available."
        )
        realtime_text   = self._sanitize_for_gemini(
            self._realtime_to_text(recent_realtime)
        )
        realtime_full   = self._sanitize_for_gemini(
            realtime_transcript or "No realtime transcript available yet."
        )

        # Scene summaries are already Gemini-generated so safer
        # but still truncate if too long
        current_text = self._sanitize_for_gemini(current_text)
        recent_text  = self._sanitize_for_gemini(recent_text)

        prompt = self._build_prompt(
            question=question,
            metadata=metadata,
            timeline=timeline,
            playback=playback,
            processed=processed,
            buffer=buffer,
            current_text=current_text,
            recent_text=recent_text,
            conversation_text=conversation_text,
            requested_text=requested_text,
            subtitle_text=subtitle_text,
            realtime_transcript=realtime_full,
            realtime_text=realtime_text,
            video_is_playing=video_is_playing,
        )

        logger.info("Prompt built — %d chars", len(prompt))

        return {
            "prompt":    prompt,
            "question":  question,
            "room_id":   room_id,
            "video_id":  video_id,
            "timestamp": playback,
        }

    # =========================================================================
    # PROMPT BUILDER — TWO MODES
    # =========================================================================

    def _build_prompt(
        self,
        *,
        question:            str,
        metadata:            Dict,
        timeline:            Dict,
        playback:            float,
        processed:           float,
        buffer:              float,
        current_text:        str,
        recent_text:         str,
        conversation_text:   str,
        requested_text:      str,
        subtitle_text:       str,
        realtime_transcript: str  = "",
        realtime_text:       str  = "",
        video_is_playing:    bool = False,
    ) -> str:

        # ── MODE 1: NO VIDEO ──────────────────────────────────────────────────
        if not video_is_playing:
            return f"""
    You are TogetherTime AI — a friendly, warm watch party companion.

    CURRENT STATE: No video is playing yet.

    ════════════════════════════════════════
    YOUR PERSONALITY & BEHAVIOR
    ════════════════════════════════════════

    - You are fun, warm, and enthusiastic — like a friend watching together
    - ALWAYS use emojis naturally in EVERY response 🎬🍿😊
    (minimum 2-3 emojis per response, more for exciting moments)
    - Match user's language and energy
    (Hindi → reply in Hindi/Hinglish with emojis)
    (Casual → be casual with emojis)
    (Excited → be excited with emojis)
    - For greetings → warm welcome with emojis 👋😄
    - For jokes → be funny and playful 😄😂
    - For video questions → enthusiastic and helpful 🎬✨
    - For general questions → friendly and knowledgeable 🧠💡
    - NEVER be robotic or dry
    - NEVER give response without at least 2 emojis
    - If user introduces themselves → remember their name 
    and use it naturally in responses

    HOW TO RESPOND:
    - Answer ANY question the user asks like a knowledgeable friend
    - For video questions → gently say no video playing yet
    - For general knowledge → answer fully and helpfully
    - For personal questions → be warm and friendly
    - NEVER say "no information available"
    - NEVER refuse to answer general knowledge questions

    CONVERSATION HISTORY:
    {conversation_text}

    USER QUESTION:
    {question}

    YOUR RESPONSE:
    """.strip()

        # ── MODE 2: VIDEO PLAYING ─────────────────────────────────────────────
        title    = metadata.get("title",   "the video")
        channel  = metadata.get("channel", "")
        current  = timeline.get("current",  "0:00")
        duration = timeline.get("duration", "0:00")
        progress = timeline.get("progressPercent", 0)

        # Calculate processing state clearly
        gap            = playback - processed
        is_caught_up   = gap <= 30.0
        processing_str = (
            f"✅ Caught up ({processed:.0f}s processed, "
            f"user at {playback:.0f}s)"
            if is_caught_up
            else
            f"⏳ Still processing (processed {processed:.0f}s, "
            f"user at {playback:.0f}s, gap={gap:.0f}s)"
        )

        return f"""
    You are TogetherTime AI — a friendly watch party companion.

    ════════════════════════════════════════
    PROCESSING STATE
    ════════════════════════════════════════

    {processing_str}

    IMPORTANT RULES BASED ON PROCESSING STATE:

    IF still processing (gap > 30s):
    - For video-specific questions (what happened, scene, dialogue, 
    summarize, what are they doing etc.)
    → Say you're still catching up BUT answer anything else normally
    - For ALL other questions (general knowledge, personal, jokes etc.)
    → Answer fully and helpfully — don't mention processing at all
    - Use your judgment — if the question needs video data say so,
    if it doesn't then just answer it

    IF caught up:
    - Answer everything normally using the context below

    ════════════════════════════════════════
    ⚠️  SPOILER RULE
    ════════════════════════════════════════

    User is at: {current} ({progress}% through)
    ONLY discuss content from BEFORE or AT {current}
    NEVER reveal anything after {current}

    ════════════════════════════════════════
    YOUR PERSONALITY
    ════════════════════════════════════════

    - Warm, friendly, conversational like a knowledgeable friend
    - Answer in same language as user (Hindi → Hindi, English → English)
    - For dialogue → give exact dialogue
    - For general knowledge → answer fully, don't say "not in context"
    - NEVER say "no information available in the provided context"
    - If question is general knowledge unrelated to video → just answer it!

    ════════════════════════════════════════
    VIDEO INFORMATION
    ════════════════════════════════════════

    Title    : {metadata.get("title",       "Unknown")}
    Channel  : {metadata.get("channel",     "Unknown")}
    Duration : {timeline.get("duration",    "Unknown")}
    Published: {metadata.get("publishedAt", "Unknown")}

    ════════════════════════════════════════
    PLAYBACK STATUS
    ════════════════════════════════════════

    Current  : {current} / {duration}
    Progress : {progress}%
    Processed: {self._seconds_to_ts(processed)}

    ════════════════════════════════════════
    CURRENT SUBTITLES
    ════════════════════════════════════════

    {subtitle_text}

    ════════════════════════════════════════
    REALTIME TRANSCRIPT (up to {current})
    ════════════════════════════════════════

    {realtime_transcript or "Not yet available"}

    ════════════════════════════════════════
    RECENT DIALOGUE
    ════════════════════════════════════════

    {realtime_text}

    ════════════════════════════════════════
    CONVERSATION HISTORY
    ════════════════════════════════════════

    {conversation_text}

    ════════════════════════════════════════
    CURRENT SCENE (at {current})
    ════════════════════════════════════════

    {current_text}

    ════════════════════════════════════════
    RECENT SCENES (before {current})
    ════════════════════════════════════════

    {recent_text}

    ════════════════════════════════════════
    USER QUESTION
    ════════════════════════════════════════

    {question}

    ════════════════════════════════════════
    YOUR ANSWER RULES:
    - If question is about the video content and you don't have it yet
    → Briefly mention you're catching up, be helpful about it
    - If question is general knowledge (PM, history, science etc.)
    → Just answer it! Don't mention video processing at all
    - If question is personal/conversational
    → Just respond naturally
    - ALWAYS be helpful, NEVER just refuse
    ════════════════════════════════════════

    YOUR ANSWER:
    """.strip()

    def _sanitize_for_gemini(self, text: str) -> str:
        """
        Clean text before including in Gemini prompt.

        Prevents PROHIBITED_CONTENT blocks caused by
        raw transcript content from videos.

        What this does:
        - Truncates very long text (main cause of blocks)
        - Removes special characters
        - Normalizes whitespace
        
        What this does NOT do:
        - Does NOT censor words
        - Does NOT change meaning
        - Does NOT remove Hindi/Odia/other languages
        """

        if not text:
            return ""

        import re

        # Truncate — shorter = less chance of triggering filters
        if len(text) > 1500:
            text = text[:1500] + "..."

        # Remove non-printable characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # =========================================================================
    # TEXT HELPERS
    # =========================================================================

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _scene_to_text(self, scene: Dict | None) -> str:
        if not scene:
            return "No scene available."

        summary    = self._clean_text(scene.get("summary",    ""))
        transcript = self._clean_text(scene.get("transcript", ""))

        return (
            f"Start: {scene.get('start')}s\n"
            f"End:   {scene.get('end')}s\n\n"
            f"Summary:\n{summary}\n\n"
            f"Transcript:\n{transcript}"
        )

    def _scenes_to_text(self, scenes: list) -> str:
        if not scenes:
            return "No recent scenes."

        blocks = []
        for index, scene in enumerate(scenes, start=1):
            blocks.append(f"Scene {index}\n{self._scene_to_text(scene)}")

        return "\n\n".join(blocks)

    def _conversation_to_text(self, conversation: list) -> str:
        if not conversation:
            return "No previous conversation."

        blocks = []
        for msg in conversation:
            role    = msg.get("role", "user").capitalize()
            content = self._clean_text(msg.get("content", ""))
            if role == "User":
                blocks.append(f"User: {content}")
            else:
                blocks.append(f"Assistant: {content}")

        return "\n".join(blocks)

    def _realtime_to_text(self, realtime_chunks: list) -> str:
        if not realtime_chunks:
            return "No recent dialogue available."

        blocks = []
        for chunk in realtime_chunks:
            start = chunk.get("start", 0)
            end   = chunk.get("end",   0)
            text  = self._clean_text(chunk.get("transcript", ""))

            if text:
                blocks.append(
                    f"[{self._seconds_to_ts(start)} → "
                    f"{self._seconds_to_ts(end)}]\n{text}"
                )

        return "\n\n".join(blocks) if blocks else "No recent dialogue available."

    def _seconds_to_ts(self, seconds: float) -> str:
        total = int(seconds)
        h     = total // 3600
        m     = (total % 3600) // 60
        s     = total % 60

        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

context_builder = ContextBuilder()