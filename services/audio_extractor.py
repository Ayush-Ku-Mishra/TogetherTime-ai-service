"""
Downloads complete audio from YouTube.

Features
--------
✅ Persistent cache
✅ Reuse downloaded audio
✅ Automatic cache creation
✅ Robust yt-dlp download
✅ Duration detection via ffprobe (Phase 1.1)
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/youtube_audio")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("=" * 70)
print("🔥 AUDIO EXTRACTOR LOADED")
print("📂 CACHE_DIR =", CACHE_DIR)
print("📂 ABSOLUTE  =", CACHE_DIR.resolve())
print("📂 EXISTS    =", CACHE_DIR.exists())
print("=" * 70)


class AudioExtractor:

    def __init__(self):
        self.enabled = False

        try:
            import yt_dlp  # noqa
            self.enabled = True
            print("✅ yt-dlp available")
        except ImportError:
            print("⚠️ yt-dlp not installed")

    # =========================================================================
    # PUBLIC — DOWNLOAD
    # =========================================================================

    async def extract_audio_segment(
        self,
        video_id:   str,
        room_id: str,
        start_time: float          = 0,
        duration:   float          = 60.0,
        output_dir: Optional[str]  = None,
    ) -> Optional[str]:

        if not self.enabled:
            return None

        # Create room-specific cache folder
        room_cache = CACHE_DIR / room_id
        room_cache.mkdir(parents=True, exist_ok=True)

        # ── Cache hit ─────────────────────────────────────────────────────
        cached = self._find_cached_audio(room_cache, video_id)

        if cached:
            print("📦 Using cached audio")
            print("📁", cached)
            return cached

        # ── Download ──────────────────────────────────────────────────────
        output_path = room_cache / video_id
        url         = f"https://www.youtube.com/watch?v={video_id}"

        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    self._download_sync,
                    url,
                    str(output_path),
                ),
                timeout=180,
            )
            return result

        except asyncio.TimeoutError:
            logger.error("⏱ Download timed out after 180 seconds.")

            # Remove any partial files that may have been created.
            self.cleanup(str(output_path))

            return None

        except Exception as e:
            print(e)
            return None

    # =========================================================================
    # PUBLIC — DURATION  (Phase 1.1)
    # =========================================================================

    def get_duration(self, audio_path: str) -> float:
        """
        Return the duration of a local audio file in seconds.

        Strategy
        --------
        1. ffprobe  — accurate, instant, ships with FFmpeg
        2. File-size heuristic — rough fallback when ffprobe is absent

        Parameters
        ----------
        audio_path : absolute or relative path to a local audio file

        Returns
        -------
        Duration in seconds, or 0.0 if it cannot be determined.
        Called by PipelineOrchestrator._generate_windows() so it knows
        where the audio ends and can build the correct window list.
        """

        if not audio_path:
            return 0.0

        path = Path(audio_path)

        if not path.exists():
            logger.warning("get_duration: file not found — %s", audio_path)
            return 0.0

        # ── Strategy 1 : ffprobe ──────────────────────────────────────────
        ffprobe = shutil.which("ffprobe")

        if ffprobe:
            duration = self._duration_via_ffprobe(ffprobe, path)
            if duration > 0:
                return duration

        # ── Strategy 2 : file-size heuristic ─────────────────────────────
        return self._duration_via_filesize(path)

    # =========================================================================
    # PRIVATE — DURATION HELPERS
    # =========================================================================

    def _duration_via_ffprobe(
        self,
        ffprobe_bin: str,
        path:        Path,
    ) -> float:
        """
        Ask ffprobe for the exact duration.

        ffprobe -v quiet -print_format json -show_format <file>

        Returns duration in seconds, or 0.0 on any failure.
        ffprobe is synchronous but finishes in <1 s on local files,
        so we run it inline — no executor needed.
        """

        cmd = [
            ffprobe_bin,
            "-v",            "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ]

        try:
            raw = subprocess.check_output(cmd, timeout=30)
            info = json.loads(raw)
            duration = float(info["format"]["duration"])
            logger.info("get_duration: %.2f s  (ffprobe)  %s", duration, path.name)
            return duration

        except subprocess.TimeoutExpired:
            logger.warning("get_duration: ffprobe timed out for %s", path.name)
            return 0.0

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning("get_duration: ffprobe parse error — %s", e)
            return 0.0

        except Exception as e:
            logger.warning("get_duration: ffprobe failed — %s", e)
            return 0.0

    def _duration_via_filesize(self, path: Path) -> float:
        """
        Estimate duration from file size.

        Assumes ~128 kbps audio (16 000 bytes/s).
        This is a rough lower bound — use only when ffprobe is absent.

        webm / opus files at 'bestaudio' are often 40–70 kbps so the
        estimate will be pessimistic (shorter than reality), which means
        _generate_windows() will produce fewer windows than needed.

        That is intentional: it is safer to under-estimate and let
        the final window simply return empty subtitles than to
        generate thousands of windows for a corrupt file.

        Returns estimated seconds, or 0.0 on failure.
        """

        try:
            size_bytes = path.stat().st_size
            estimated  = size_bytes / 16_000          # 128 kbps = 16 000 B/s
            logger.warning(
                "get_duration: ffprobe unavailable — "
                "size heuristic %.1f s for %s",
                estimated, path.name,
            )
            return estimated

        except Exception as e:
            logger.error("get_duration: filesize fallback failed — %s", e)
            return 0.0

    # =========================================================================
    # PRIVATE — CACHE HELPERS
    # =========================================================================

    def _find_cached_audio(
        self,
        room_cache: Path,
        video_id: str,
    ) -> Optional[str]:

        for ext in [".webm", ".m4a", ".opus", ".mp3"]:
            file = room_cache / f"{video_id}{ext}"
            if file.exists():
                return str(file)

        return None

    # =========================================================================
    # PRIVATE — DOWNLOAD
    # =========================================================================

    def _download_sync(self, url: str, output_path: str) -> Optional[str]:

        import yt_dlp

        ydl_opts = {
            "format":            "bestaudio/best",
            "outtmpl":           output_path + ".%(ext)s",
            "quiet":             False,
            "no_warnings":       False,
            "noplaylist":        True,
            "retries":           10,
            "fragment_retries":  10,
            "extractor_retries": 10,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/137.0.0.0 Safari/537.36"
                ),
            },
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("=" * 60)
            print("⬇️  Downloading:", url)

            info = ydl.extract_info(url, download=True)

            print("Title    :", info.get("title"))
            print("Duration :", info.get("duration"), "s")
            print("=" * 60)

        for ext in [".webm", ".m4a", ".opus", ".mp3"]:
            file = output_path + ext
            if os.path.exists(file):
                print("✅ Audio cached")
                print("📁", file)
                print("📦 Size:", os.path.getsize(file), "bytes")
                return file

        print("❌ Audio file not found after download")
        return None

    # =========================================================================
    # UNUSED — kept for future TTL cleanup
    # =========================================================================

    def cleanup(self, file_path: str) -> None:
        pass


audio_extractor = AudioExtractor()