"""
YouTube Service

Fetches video metadata from YouTube.

One function does everything:
    get_youtube_details()  →  title, channel, description,
                              publishedAt, thumbnail, duration

Strategy
--------
1. YouTube Data API v3  (requires API key, returns full metadata)
2. oEmbed fallback      (no key needed, returns title + channel only)

Duration format
---------------
YouTube returns ISO 8601 duration strings:
    PT2M28S   →  2 minutes 28 seconds
    PT1H42M   →  1 hour 42 minutes
    PT58S     →  58 seconds

Raw string is returned here.
Call parse_iso_duration() elsewhere to convert to seconds.
"""

import httpx
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_KEY   = os.getenv("TMDB_API_KEY", "")
YT_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# ── Shared client config ──────────────────────────────────────────────────────
_CLIENT_KWARGS = {
    "timeout":          10.0,
    "verify":           False,   # Fixes Windows SSL/firewall issues
    "follow_redirects": True,
}

# ── Empty result shape ────────────────────────────────────────────────────────
_EMPTY = {
    "title":       "",
    "channel":     "",
    "description": "",
    "publishedAt": "",
    "thumbnail":   "",
    "duration":    "",   # raw ISO 8601 string e.g. "PT2M28S"
}


# ─────────────────────────────────────────────────────────────────────────────
# get_youtube_details  (single entry point for all metadata)
# ─────────────────────────────────────────────────────────────────────────────

async def get_youtube_details(video_id: str) -> dict:
    """
    Return all available metadata for a YouTube video in one call.

    Returns
    -------
    {
        "title":       str,   human-readable video title
        "channel":     str,   channel / uploader name
        "description": str,   full video description
        "publishedAt": str,   ISO 8601 publish date  e.g. "2024-01-15T10:30:00Z"
        "thumbnail":   str,   URL of high-quality thumbnail
        "duration":    str,   ISO 8601 duration      e.g. "PT1H42M11S"
    }

    Falls back to oEmbed when no API key is available
    (description, publishedAt, thumbnail, duration will be empty strings).
    """

    if not video_id:
        return dict(_EMPTY)

    # ── Method 1 : YouTube Data API v3 ───────────────────────────────────
    if YT_API_KEY:
        result = await _fetch_via_data_api(video_id)
        if result:
            return result

    # ── Method 2 : oEmbed fallback ────────────────────────────────────────
    result = await _fetch_via_oembed(video_id)
    if result:
        return result

    # ── Nothing worked ────────────────────────────────────────────────────
    print(f"⚠️ Could not fetch metadata for: {video_id}")
    return dict(_EMPTY)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_via_data_api(video_id: str) -> dict | None:
    """
    Fetch full metadata using YouTube Data API v3.

    Requires YOUTUBE_API_KEY environment variable.

    Parts requested
    ---------------
    snippet        → title, channel, description, publishedAt, thumbnail
    contentDetails → duration (ISO 8601 string)

    Returns None on any failure so the caller can fall back.
    """

    try:
        async with httpx.AsyncClient(**_CLIENT_KWARGS) as client:

            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part":   "snippet,contentDetails",
                    "id":     video_id,
                    "key":    YT_API_KEY,
                    "fields": (
                        "items("
                        "snippet(title,channelTitle,description,"
                        "publishedAt,thumbnails/high/url),"
                        "contentDetails/duration"
                        ")"
                    ),
                },
            )

            data  = resp.json()
            items = data.get("items", [])

            if not items:
                print(f"⚠️ YouTube API: no items found for {video_id}")
                return None

            snippet = items[0].get("snippet",        {})
            content = items[0].get("contentDetails", {})

            result = {
                "title":       snippet.get("title",       ""),
                "channel":     snippet.get("channelTitle", ""),
                "description": snippet.get("description",  ""),
                "publishedAt": snippet.get("publishedAt",  ""),
                "thumbnail":   (
                    snippet
                    .get("thumbnails", {})
                    .get("high",       {})
                    .get("url",        "")
                ),
                "duration":    content.get("duration", ""),
            }

            print(
                f"✅ YouTube API: {result['title']} "
                f"| duration={result['duration']}"
            )

            return result

    except Exception as e:
        print(f"⚠️ YouTube Data API failed: {type(e).__name__}: {e}")
        return None


async def _fetch_via_oembed(video_id: str) -> dict | None:
    """
    Fetch basic metadata via YouTube oEmbed (no API key required).

    oEmbed only returns title and author — description, publishedAt,
    thumbnail URL (high quality) and duration will be empty strings.

    Returns None on any failure.
    """

    try:
        async with httpx.AsyncClient(**_CLIENT_KWARGS) as client:

            resp = await client.get(
                "https://www.youtube.com/oembed",
                params={
                    "url":    f"https://www.youtube.com/watch?v={video_id}",
                    "format": "json",
                },
            )

            if resp.status_code != 200:
                return None

            data = resp.json()

            result = {
                "title":       data.get("title",       ""),
                "channel":     data.get("author_name", ""),
                "description": "",
                "publishedAt": "",
                "thumbnail":   data.get("thumbnail_url", ""),
                "duration":    "",
            }

            print(f"✅ YouTube oEmbed: {result['title']}")

            return result

    except Exception as e:
        print(f"⚠️ YouTube oEmbed failed: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Duration parser  (reusable utility)
# ─────────────────────────────────────────────────────────────────────────────

def parse_iso_duration(iso: str) -> int:
    """
    Convert an ISO 8601 duration string to total seconds.

    Examples
    --------
    "PT2M28S"    →  148
    "PT1H42M11S" →  6131
    "PT58S"      →   58
    "PT1H"       → 3600
    ""           →    0

    Parameters
    ----------
    iso : raw duration string from YouTube API

    Returns
    -------
    Total duration in seconds (int), or 0 if parsing fails.
    """

    if not iso:
        return 0

    import re

    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match   = re.fullmatch(pattern, iso)

    if not match:
        print(f"⚠️ parse_iso_duration: unrecognised format '{iso}'")
        return 0

    hours   = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    total = hours * 3600 + minutes * 60 + seconds

    return total


# ─────────────────────────────────────────────────────────────────────────────
# Platform router  (kept for backwards compatibility)
# ─────────────────────────────────────────────────────────────────────────────

async def get_video_title(platform: str, video_id: str) -> str:
    """
    Return a human-readable title for any supported platform.

    YouTube uses get_youtube_details() internally —
    no separate title-only API call is made.
    """

    if not video_id:
        return "Unknown content"

    platform = (platform or "").lower()

    if platform == "youtube":
        details = await get_youtube_details(video_id)
        title   = details["title"]
        channel = details["channel"]
        return f"{title} by {channel}" if channel else title

    elif platform == "vimeo":
        try:
            async with httpx.AsyncClient(**_CLIENT_KWARGS) as client:
                resp = await client.get(
                    f"https://vimeo.com/api/v2/video/{video_id}.json"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data[0].get("title", f"Vimeo video ({video_id})")
        except Exception as e:
            print(f"⚠️ Vimeo title fetch failed: {e}")
        return f"Vimeo video ({video_id})"

    elif platform == "twitch":
        return f"Twitch stream ({video_id})"

    elif platform == "dailymotion":
        try:
            async with httpx.AsyncClient(**_CLIENT_KWARGS) as client:
                resp = await client.get(
                    f"https://api.dailymotion.com/video/{video_id}?fields=title"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("title", f"Dailymotion video ({video_id})")
        except Exception as e:
            print(f"⚠️ Dailymotion title fetch failed: {e}")
        return f"Dailymotion video ({video_id})"

    return f"{platform} video ({video_id})"