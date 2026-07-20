"""
Video Metadata Service

Single source of truth for video metadata.

Every video that passes through the pipeline has metadata.
This service fetches it, normalises it, and returns one
clean dictionary that every other service can consume.

Supported platforms
-------------------
YouTube    ✅  (title, channel, description, duration, publishedAt, thumbnail)
Vimeo      🔜  (title only for now)
Twitch     🔜  (title only for now)
Dailymotion 🔜 (title only for now)

Flow
----
Caller
  │
  ▼
get_metadata(platform, video_id)
  │
  ▼
_fetch_youtube()  /  _fetch_other()
  │
  ▼
Normalised dict
  │
  ▼
Returned to caller

Does NOT
--------
❌ Store to Redis
❌ Store to Qdrant
❌ Build prompts
❌ Call Gemini
❌ Call Whisper

It only fetches and normalises metadata.
"""

from __future__ import annotations

import logging
from typing import Dict

from .youtube_service import get_youtube_details, get_video_title
from .state_persistence import state_persistence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Empty / default shape
# Every key is always present so callers never get KeyError.
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY_METADATA: Dict = {
    "videoId":     "",
    "platform":    "",
    "title":       "",
    "channel":     "",
    "description": "",
    "duration":    "",   # raw ISO 8601 string  e.g. "PT1H42M11S"
    "publishedAt": "",   # ISO 8601 date        e.g. "2024-01-15T10:30:00Z"
    "thumbnail":   "",   # URL string
}


# ─────────────────────────────────────────────────────────────────────────────
# VideoMetadataService
# ─────────────────────────────────────────────────────────────────────────────

class VideoMetadataService:
    """
    Single entry point for video metadata across all platforms.

    Usage
    -----
    from services.video_metadata import video_metadata

    meta = await video_metadata.get_metadata(
        platform="youtube",
        video_id="dQw4w9WgXcQ",
    )

    print(meta["title"])
    print(meta["duration"])   # "PT3M33S"
    """

    def __init__(self):
        self._cache: Dict[str, Dict] = {}    # simple in-memory cache
        logger.info("✅ VideoMetadataService initialized")

    async def get_metadata(
        self,
        platform: str,
        video_id: str,
    ) -> Dict:

        cache_key = f"{platform}:{video_id}"

        # ── Check in-memory cache first ───────────────────────────────────────
        if cache_key in self._cache:
            logger.debug("get_metadata: memory cache hit — %s", cache_key)
            return self._cache[cache_key]

        # ── Check Redis cache ─────────────────────────────────────────────────
        redis_cached = state_persistence.load_metadata(platform, video_id)
        if redis_cached:
            self._cache[cache_key] = redis_cached
            logger.info(
                "get_metadata: Redis cache hit — %s ✅", cache_key
            )
            return redis_cached

        if not video_id:
            return self._empty(platform, video_id)

        platform = (platform or "youtube").lower().strip()

        try:
            if platform == "youtube":
                result = await self._fetch_youtube(video_id)
            else:
                result = await self._fetch_other(platform, video_id)

            # ── Save to both caches ───────────────────────────────────────────
            self._cache[cache_key] = result
            state_persistence.save_metadata(platform, video_id, result)

            return result

        except Exception:
            logger.exception("get_metadata failed")
            return self._empty(platform, video_id)

    # =========================================================================
    # PRIVATE — YOUTUBE
    # =========================================================================

    async def _fetch_youtube(self, video_id: str) -> Dict:
        """
        Fetch YouTube metadata via get_youtube_details().

        get_youtube_details() already handles:
        - YouTube Data API v3 (full metadata)
        - oEmbed fallback     (title + channel only)

        We just normalise the result into our standard shape.
        """

        details = await get_youtube_details(video_id)

        metadata = {
            "videoId":     video_id,
            "platform":    "youtube",
            "title":       details.get("title",       ""),
            "channel":     details.get("channel",     ""),
            "description": details.get("description", ""),
            "duration":    details.get("duration",    ""),
            "publishedAt": details.get("publishedAt", ""),
            "thumbnail":   details.get("thumbnail",   ""),
        }

        logger.info(
            "YouTube metadata — title=%s  duration=%s  channel=%s",
            metadata["title"]    or "(none)",
            metadata["duration"] or "(none)",
            metadata["channel"]  or "(none)",
        )

        return metadata

    # =========================================================================
    # PRIVATE — OTHER PLATFORMS
    # =========================================================================

    async def _fetch_other(self, platform: str, video_id: str) -> Dict:
        """
        Fetch metadata for non-YouTube platforms.

        Currently only retrieves title via get_video_title().
        Other fields (duration, description, etc.) are left empty
        until platform-specific APIs are integrated.
        """

        title = await get_video_title(platform, video_id)

        metadata = {
            **self._empty(platform, video_id),
            "title": title,
        }

        logger.info(
            "%s metadata — title=%s",
            platform, title or "(none)",
        )

        return metadata

    # =========================================================================
    # PRIVATE — EMPTY SHAPE
    # =========================================================================

    def _empty(self, platform: str, video_id: str) -> Dict:
        """
        Return the default empty metadata dict with platform/videoId filled.

        Used when fetch fails or video_id is missing.
        All other callers always get a dict with the correct keys.
        """

        return {
            **_EMPTY_METADATA,
            "videoId":  video_id,
            "platform": platform,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

video_metadata = VideoMetadataService()