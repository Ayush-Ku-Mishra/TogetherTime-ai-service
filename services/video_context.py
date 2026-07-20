"""
Video Context Service
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .video_metadata      import video_metadata
from .processing_queue    import processing_queue
from .qdrant_service      import qdrant_service
from .timeline_service    import timeline_service
from .conversation_memory import conversation_memory
from .subtitle_service    import subtitle_service
from .spoiler_guard       import spoiler_guard
from .scene_navigator     import scene_navigator
from .youtube_service     import parse_iso_duration

logger = logging.getLogger(__name__)


class VideoContextService:

    def __init__(self):
        logger.info("✅ VideoContextService initialized")

    async def build(
        self,
        *,
        room_id:           str,
        video_id:          str,
        platform:          str,
        question:          str   = "",
        current_timestamp: float = 0.0,
    ) -> Dict:

        # ── Metadata ──────────────────────────────────────────────────────────
        metadata = await video_metadata.get_metadata(
            platform=platform,
            video_id=video_id,
        )

        # ── Playback position ─────────────────────────────────────────────────
        queue_playback = processing_queue.get_current_playback(
            room_id=room_id,
            video_id=video_id,
        )

        current_playback = (
            current_timestamp
            if current_timestamp > 0
            else queue_playback
        )

        # ── Processed until ───────────────────────────────────────────────────
        processed_until = processing_queue.get_processed_until(
            room_id=room_id,
            video_id=video_id,
        )


        # ── Rest of build() stays exactly the same ────────────────────────────
        buffer_ahead = processed_until - current_playback

        # ── Timeline ──────────────────────────────────────────────────────
        duration_iso = metadata.get("duration", "")
        duration_sec = parse_iso_duration(duration_iso) if duration_iso else 0.0

        timeline = timeline_service.get_progress_info(
            current_playback=current_playback,
            duration_seconds=duration_sec,
        )

        # ── Current scene — FILTERED by current_playback ──────────────────
        # Only get scene that is actually within user's watched range
        # A scene is valid if it STARTS before current_playback
        raw_current_scene = await qdrant_service.get_current_scene(
            room_id=room_id,
            video_id=video_id,
            timestamp=current_playback,
        )

        # Filter: reject scene if it starts ahead of current playback
        current_scene = self._filter_scene(raw_current_scene, current_playback)

        # ── Recent scenes — FILTERED by current_playback ──────────────────
        raw_recent_scenes = await qdrant_service.get_recent_scenes(
            room_id=room_id,
            video_id=video_id,
            before=current_playback,
            limit=5,
        )

        # Extra safety filter — remove any scene that ends after playback
        recent_scenes = [
            s for s in raw_recent_scenes
            if s.get("start", 0) <= current_playback
        ]

        # ── Conversation history ──────────────────────────────────────────
        conversation_history = conversation_memory.get_history(
            room_id=room_id,
            limit=10,
        )

        # ── Subtitle history ──────────────────────────────────────────────
        subtitle_history = await subtitle_service.get_last_subtitles_with_fallback(
            room_id=room_id,
            video_id=video_id,
            limit=10,
        )

        # ── Current subtitle ──────────────────────────────────────────────
        current_subs = subtitle_service.get_current_subtitles(
            room_id=room_id,
            video_id=video_id,
            current_playback=current_playback,
            window_seconds=5.0,
        )

        current_subtitle = (
            subtitle_service.format_for_prompt(current_subs, max_chars=300)
            if current_subs else ""
        )

        # ── Requested scene ───────────────────────────────────────────────
        requested_scene: Optional[Dict] = None

        if question:
            has_time_reference = (
                timeline_service.parse_timestamp_from_question(
                    question=question,
                    current_playback=current_playback,
                    duration_seconds=duration_sec,
                ) is not None
            )

            nav_keywords = [
                "beginning", "first",
                "previous", "back",
                "next scene", "after",
                "ago",
                "at 0:", "at 1:", "at 2:", "at 3:",
            ]

            non_nav_keywords = [
                "recommend", "similar", "like this",
                "suggest", "summary", "summarize", "recap",
                "ending", "end", "last", "final",
            ]

            has_nav_keyword = (
                any(kw in question.lower() for kw in nav_keywords)
                and not any(kw in question.lower() for kw in non_nav_keywords)
            )

            if has_time_reference or has_nav_keyword:
                raw_requested = await scene_navigator.navigate(
                    question=question,
                    room_id=room_id,
                    video_id=video_id,
                    current_playback=current_playback,
                    duration_seconds=duration_sec,
                )
                # Filter requested scene too
                requested_scene = self._filter_scene(raw_requested, current_playback)

        # ── Realtime Memory — FILTERED by current_playback ────────────────
        try:
            from .realtime_memory import realtime_memory

            # Only get realtime transcript up to current_playback
            # NOT up to processed_until
            realtime_transcript = realtime_memory.get_transcript_at(
                room_id=room_id,
                video_id=video_id,
                timestamp=current_playback,
                window=30.0,
            )

            # Filter recent realtime chunks by current_playback
            raw_realtime = realtime_memory.get_recent_transcripts(
                room_id=room_id,
                video_id=video_id,
                before=current_playback,
                limit=3,
            )

            # Extra safety — reject chunks that start after current_playback
            recent_realtime = [
                chunk for chunk in raw_realtime
                if chunk.get("start", 0) <= current_playback
            ]

            transcript_ready = realtime_memory.get_transcript_ready(
                room_id=room_id,
                video_id=video_id,
            )

            is_transcript_ready = realtime_memory.is_ready_for_question(
                room_id=room_id,
                video_id=video_id,
                current_timestamp=current_playback,
                buffer_seconds=10.0,
            )

        except Exception:
            logger.warning("RealtimeMemory unavailable — room=%s", room_id)
            realtime_transcript = ""
            recent_realtime     = []
            transcript_ready    = 0.0
            is_transcript_ready = False

        # ── Spoiler guard ─────────────────────────────────────────────────
        spoiler_result = spoiler_guard.check(
            question=question,
            current_playback=current_playback,
            processed_until=processed_until,
            duration_seconds=duration_sec,
        )

        spoiler_allowed = spoiler_result["allowed"]

        # ── Debug print ───────────────────────────────────────────────────
        print("=" * 60)
        print("VIDEO CONTEXT")
        print(f"Playback    : {current_playback:.1f}s  ({timeline['current']})")
        print(f"Processed   : {processed_until:.1f}s")
        print(f"Duration    : {duration_sec:.1f}s  ({timeline['duration']})")
        print(f"Remaining   : {timeline['remaining']}")
        print(f"Progress    : {timeline['progressPercent']}%")
        print(f"Buffer      : {buffer_ahead:.1f}s")
        print(f"CurrScene   : {current_scene.get('start') if current_scene else 'None'} → {current_scene.get('end') if current_scene else 'None'}")
        print(f"RecentScenes: {len(recent_scenes)}")
        print(f"ReqScene    : {requested_scene.get('start') if requested_scene else 'None'} → {requested_scene.get('end') if requested_scene else 'None'}")
        print(f"Conversation: {len(conversation_history)} message(s)")
        print(f"Subtitles   : {len(subtitle_history)} recent")
        print(f"CurrSub     : {current_subtitle[:60] if current_subtitle else 'None'}")
        print(f"RTTranscript: {transcript_ready:.1f}s ready")
        print(f"RTReady     : {is_transcript_ready}")
        print(f"Spoiler OK  : {spoiler_allowed}")
        if not spoiler_allowed:
            print(f"Blocked     : {spoiler_result['reason']}")
        print("=" * 60)

        # ── Assemble context ──────────────────────────────────────────────
        context: Dict = {
            "roomId":   room_id,
            "videoId":  video_id,
            "platform": platform,
            "metadata": metadata,
            "timeline":        timeline,
            "currentPlayback": current_playback,
            "processedUntil":  processed_until,
            "bufferAhead":     round(buffer_ahead, 1),
            "currentScene":    current_scene,
            "recentScenes":    recent_scenes,
            "relatedScenes":   [],
            "requestedScene":  requested_scene,
            "conversationHistory": conversation_history,
            "subtitleHistory":     subtitle_history,
            "currentSubtitle":     current_subtitle,
            "realtimeTranscript":  realtime_transcript or "",
            "recentRealtime":      recent_realtime     or [],
            "transcriptReady":     transcript_ready    or 0.0,
            "isTranscriptReady":   is_transcript_ready or False,
            "spoilerAllowed":  spoiler_allowed,
            "spoilerMessage":  spoiler_result.get("message", ""),
            "intent":          {},
        }

        logger.info(
            "build done — playback=%s  duration=%s  progress=%.1f%%  "
            "scenes(curr=%s req=%s recent=%d)  rt_ready=%.1fs  spoiler=%s",
            timeline["current"],
            timeline["duration"],
            timeline["progressPercent"],
            "YES" if current_scene   else "NO",
            "YES" if requested_scene else "NO",
            len(recent_scenes),
            transcript_ready,
            "ALLOW" if spoiler_allowed else "BLOCK",
        )

        return context

    # =========================================================================
    # PRIVATE — SCENE FILTER
    # =========================================================================

    def _filter_scene(
        self,
        scene:            Optional[Dict],
        current_playback: float,
        tolerance:        float = 10.0,
    ) -> Optional[Dict]:
        """
        Reject any scene that starts significantly
        ahead of current playback.

        A scene is valid if:
            scene.start <= current_playback + tolerance

        The tolerance (10s) allows for slight timing mismatches
        between playback tracking and scene boundaries.

        Returns None if scene is from the future.
        Returns scene unchanged if it is within the watched range.
        """

        if not scene:
            return None

        scene_start = scene.get("start", 0)

        if scene_start > current_playback + tolerance:
            logger.info(
                "_filter_scene: rejected future scene  "
                "scene_start=%.1fs > playback=%.1fs + tolerance=%.1fs",
                scene_start, current_playback, tolerance,
            )
            return None

        return scene


video_context = VideoContextService()