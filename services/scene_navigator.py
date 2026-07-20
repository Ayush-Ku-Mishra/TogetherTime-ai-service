"""
Scene Navigator

Answers "Which scene do I need?"

Given a user request, returns the most relevant scene from storage.

Examples
--------
"What happened at 1:15?"   → scene containing timestamp 75s
"Previous scene"           → scene before current playback
"Next scene"               → scene after current playback
"Beginning"                → first stored scene
"Ending"                   → last stored scene

Does NOT
--------
❌ Call Gemini
❌ Build prompts
❌ Store anything

Only navigates scenes in storage.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .qdrant_service   import qdrant_service
from .timeline_service import timeline_service

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SceneNavigator
# ─────────────────────────────────────────────────────────────────────────────

class SceneNavigator:
    """
    Finds the right scene for any temporal navigation request.
    """

    def __init__(self):
        logger.info("✅ SceneNavigator initialized")

    # =========================================================================
    # PUBLIC — GET SCENE AT TIMESTAMP
    # =========================================================================

    async def get_scene_at(
        self,
        room_id:   str,
        video_id:  str,
        timestamp: float,
    ) -> Optional[Dict]:
        """
        Return the scene whose time range contains the timestamp.

        Parameters
        ----------
        timestamp : absolute timestamp in seconds

        Returns
        -------
        Scene dict, or None if not found.
        """

        scene = await qdrant_service.get_current_scene(
            room_id=room_id,
            video_id=video_id,
            timestamp=timestamp,
        )

        ts_str = timeline_service.seconds_to_hhmmss(timestamp)

        logger.info(
            "get_scene_at — room=%s  timestamp=%s  found=%s",
            room_id, ts_str, "YES" if scene else "NO",
        )

        return scene

    # =========================================================================
    # PUBLIC — GET PREVIOUS SCENE
    # =========================================================================

    async def get_previous_scene(
        self,
        room_id:          str,
        video_id:         str,
        current_playback: float,
    ) -> Optional[Dict]:
        """
        Return the scene immediately before current playback.

        Parameters
        ----------
        current_playback : current position in seconds

        Returns
        -------
        Previous scene dict, or None if at the beginning.
        """

        all_scenes = await qdrant_service.get_video_scenes(
            room_id=room_id,
            video_id=video_id,
        )

        if not all_scenes:
            return None

        # Sort chronologically
        sorted_scenes = sorted(
            all_scenes,
            key=lambda s: s.get("start", 0),
        )

        # Find scenes that END before current playback
        previous = [
            s for s in sorted_scenes
            if s.get("end", 0) < current_playback - 1
        ]

        if not previous:
            logger.info(
                "get_previous_scene — room=%s  none found (at beginning)",
                room_id,
            )
            return None

        scene = previous[-1]   # Most recent one before current

        logger.info(
            "get_previous_scene — room=%s  found %.1fs→%.1fs",
            room_id,
            scene.get("start", 0),
            scene.get("end",   0),
        )

        return scene

    # =========================================================================
    # PUBLIC — GET NEXT SCENE
    # =========================================================================

    async def get_next_scene(
        self,
        room_id:          str,
        video_id:         str,
        current_playback: float,
    ) -> Optional[Dict]:
        """
        Return the scene immediately after current playback.

        Parameters
        ----------
        current_playback : current position in seconds

        Returns
        -------
        Next scene dict, or None if at the end.
        """

        all_scenes = await qdrant_service.get_video_scenes(
            room_id=room_id,
            video_id=video_id,
        )

        if not all_scenes:
            return None

        sorted_scenes = sorted(
            all_scenes,
            key=lambda s: s.get("start", 0),
        )

        # Find scenes that START after current playback
        future = [
            s for s in sorted_scenes
            if s.get("start", 0) > current_playback + 1
        ]

        if not future:
            logger.info(
                "get_next_scene — room=%s  none found (at end)",
                room_id,
            )
            return None

        scene = future[0]   # First one after current

        logger.info(
            "get_next_scene — room=%s  found %.1fs→%.1fs",
            room_id,
            scene.get("start", 0),
            scene.get("end",   0),
        )

        return scene

    # =========================================================================
    # PUBLIC — GET FIRST SCENE
    # =========================================================================

    async def get_first_scene(
        self,
        room_id:  str,
        video_id: str,
    ) -> Optional[Dict]:
        """Return the first stored scene."""

        all_scenes = await qdrant_service.get_video_scenes(
            room_id=room_id,
            video_id=video_id,
        )

        if not all_scenes:
            return None

        scene = min(all_scenes, key=lambda s: s.get("start", 0))

        logger.info(
            "get_first_scene — room=%s  found %.1fs→%.1fs",
            room_id,
            scene.get("start", 0),
            scene.get("end",   0),
        )

        return scene

    # =========================================================================
    # PUBLIC — GET LAST SCENE
    # =========================================================================

    async def get_last_scene(
        self,
        room_id:  str,
        video_id: str,
    ) -> Optional[Dict]:
        """Return the last stored scene."""

        all_scenes = await qdrant_service.get_video_scenes(
            room_id=room_id,
            video_id=video_id,
        )

        if not all_scenes:
            return None

        scene = max(all_scenes, key=lambda s: s.get("start", 0))

        logger.info(
            "get_last_scene — room=%s  found %.1fs→%.1fs",
            room_id,
            scene.get("start", 0),
            scene.get("end",   0),
        )

        return scene

    # =========================================================================
    # PUBLIC — NAVIGATE FROM QUESTION
    # =========================================================================

    async def navigate(
        self,
        question:         str,
        room_id:          str,
        video_id:         str,
        current_playback: float,
        duration_seconds: float,
    ) -> Optional[Dict]:
        """
        Automatically choose the correct scene based on the question.

        Examples
        --------
        "What happened at 1:15?"   → get_scene_at(75)
        "Go to the beginning"      → get_first_scene()
        "Show me the ending"       → get_last_scene()
        "Previous scene"           → get_previous_scene()
        "Next scene"               → get_next_scene()
        None of the above          → get_scene_at(current_playback)
        """

        q = question.lower().strip()

        # ── Check for explicit timestamp ──────────────────────────────────
        requested_ts = timeline_service.parse_timestamp_from_question(
            question=question,
            current_playback=current_playback,
            duration_seconds=duration_seconds,
        )

        if requested_ts is not None:
            logger.info(
                "navigate — explicit timestamp %.1fs", requested_ts
            )
            return await self.get_scene_at(room_id, video_id, requested_ts)

        # ── Check for navigation keywords ─────────────────────────────────
        if any(kw in q for kw in ["beginning", "start", "first", "from the start"]):
            return await self.get_first_scene(room_id, video_id)

        if any(kw in q for kw in ["ending", "end", "last", "final"]):
            return await self.get_last_scene(room_id, video_id)

        if any(kw in q for kw in ["previous", "before", "back", "last scene"]):
            return await self.get_previous_scene(room_id, video_id, current_playback)

        if any(kw in q for kw in ["next", "after", "following", "next scene"]):
            return await self.get_next_scene(room_id, video_id, current_playback)

        # ── Default: current position ─────────────────────────────────────
        return await self.get_scene_at(room_id, video_id, current_playback)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

scene_navigator = SceneNavigator()