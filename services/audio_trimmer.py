"""
Cuts a short audio segment from a cached YouTube audio file.

Uses FFmpeg stream copy (no re-encoding)
which makes trimming extremely fast.

Flow
----
Cached Audio
      │
      ▼
 trim_audio()
      │
      ▼
 asyncio.to_thread()
      │
      ▼
 _trim_sync()
      │
      ▼
 subprocess.run(ffmpeg)
      │
      ▼
Temporary Clip
"""

import logging
import shutil
import subprocess
import asyncio
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

CLIP_CACHE = Path("cache/audio_clips")
CLIP_CACHE.mkdir(parents=True, exist_ok=True)


class AudioTrimmer:

    def __init__(self):

        self.enabled = shutil.which("ffmpeg") is not None

        if self.enabled:
            print("✅ FFmpeg detected")
        else:
            print("⚠️ FFmpeg not found in PATH")

    # =========================================================================
    # PUBLIC — ASYNC ENTRY POINT
    # =========================================================================

    async def trim_audio(
        self,
        input_file: str,
        start_time: float,
        duration:   float = 60.0,
    ) -> Optional[str]:

        if not self.enabled:
            return None

        input_path = Path(input_file)

        if not input_path.exists():
            logger.warning("Audio file not found: %s", input_file)
            return None

        # ── Fix 1: reject windows that start at or past EOF ───────────────────
        # FFmpeg will produce an empty WAV → Whisper crashes with
        # "ValueError: max() arg is an empty sequence"
        try:
            import subprocess, shutil, json
            ffprobe = shutil.which("ffprobe")
            if ffprobe:
                raw = subprocess.check_output([
                    ffprobe, "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    str(input_path),
                ], timeout=10)
                info = json.loads(raw)
                file_duration = float(info["format"]["duration"])

                if start_time >= file_duration - 0.5:
                    logger.info(
                        "trim_audio: start_time=%.3fs >= file_duration=%.3fs "
                        "— nothing to trim, skipping",
                        start_time, file_duration,
                    )
                    return None

                # Clamp duration so we never read past EOF
                duration = min(duration, file_duration - start_time)

                # ── Fix 2: skip if remaining audio < 1 second ─────────────────
                if duration < 1.0:
                    logger.info(
                        "trim_audio: clamped duration=%.3fs < 1.0s — skipping",
                        duration,
                    )
                    return None

        except Exception as e:
            logger.warning("trim_audio: ffprobe pre-check failed — %s", e)
            # Continue anyway — FFmpeg itself will handle the edge case

        output_file = (
            CLIP_CACHE
            / f"{input_path.stem}_{int(start_time)}_{int(duration)}.wav"
        )

        # ── Cache hit — reuse existing clip ───────────────────────────────────
        if output_file.exists():
            # ── Fix 3: don't reuse empty cached clips ─────────────────────────
            if output_file.stat().st_size < 1000:
                logger.warning(
                    "trim_audio: cached clip too small (%d bytes) — re-trimming",
                    output_file.stat().st_size,
                )
                output_file.unlink(missing_ok=True)
            else:
                logger.info("📦 Using cached clip: %s", output_file.name)
                return str(output_file)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",   str(start_time),
            "-i",    str(input_path),
            "-t",    str(duration),
            "-vn",
            "-ac",   "1",
            "-ar",   "16000",
            "-c:a",  "pcm_s16le",
            str(output_file),
        ]

        logger.info(
            "✂️ Trimming  %.1fs → %.1fs  from %s",
            start_time,
            start_time + duration,
            input_path.name,
        )

        result = await asyncio.to_thread(
            self._trim_sync,
            cmd,
            output_file,
        )

        return result

    # =========================================================================
    # PRIVATE — SYNCHRONOUS FFMPEG CALL
    # =========================================================================

    def _trim_sync(
        self,
        cmd:         list,
        output_file: Path,
    ) -> Optional[str]:

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.error("❌ FFmpeg failed (code %d)", result.returncode)
                logger.error(result.stderr)
                return None

            if not output_file.exists():
                logger.error(
                    "❌ FFmpeg returned 0 but output file missing: %s",
                    output_file,
                )
                return None

            # ── Fix 4: validate output file is not empty ──────────────────────
            file_size = output_file.stat().st_size
            if file_size < 1000:
                logger.warning(
                    "❌ FFmpeg output too small (%d bytes) — "
                    "likely empty clip at EOF, discarding",
                    file_size,
                )
                output_file.unlink(missing_ok=True)
                return None

            logger.info("✂️ Clip created: %s  (%d bytes)", output_file.name, file_size)
            return str(output_file)

        except subprocess.TimeoutExpired:
            logger.error("⏱ FFmpeg timeout after 120s")
            return None

        except Exception as e:
            logger.exception("FFmpeg error: %s", e)
            return None

    # =========================================================================
    # PUBLIC — CLEANUP
    # =========================================================================

    def cleanup(self, clip_path: str) -> None:
        """
        Delete a temporary clip created by trim_audio().

        The orchestrator is responsible for calling this after every
        window, whether transcription succeeded or not, so temp files
        do not accumulate on disk.

        Parameters
        ----------
        clip_path : absolute path returned by trim_audio()
        """
        try:
            Path(clip_path).unlink(missing_ok=True)
            logger.debug("Deleted clip: %s", clip_path)
        except Exception as e:
            logger.warning("cleanup failed for %s: %s", clip_path, e)


audio_trimmer = AudioTrimmer()