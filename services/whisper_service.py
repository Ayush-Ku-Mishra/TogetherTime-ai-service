"""
Faster Whisper transcription service.
Handles multilingual audio including Odia, Hindi, etc.
"""

import asyncio
import os
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")


class WhisperService:

    def __init__(self):
        self.model = None
        self.enabled = False

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            self.enabled = True
            print(f"✅ Faster Whisper loaded: {WHISPER_MODEL_SIZE}")

        except ImportError:
            print("⚠️ faster-whisper not installed")

        except Exception as e:
            print(f"⚠️ Whisper model load failed: {e}")

    def ensure_loaded(self):
        """
        Load Whisper only when first needed.
        """
        if self.model is None:
            self._load_model()

    # ── Main Transcription ─────────────────────────────────────────────────

    async def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        current_time_offset: float = 0.0,
    ) -> List[Dict]:

        self.ensure_loaded()

        if not self.enabled or not self.model:
            return []

        if not os.path.exists(audio_path):
            print(f"⚠️ Audio file not found: {audio_path}")
            return []

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None,
                self._transcribe_sync,
                audio_path,
                language,
                current_time_offset,
            )

            return result

        except Exception as e:
            print(f"⚠️ Whisper transcription error: {e}")
            return []

    def _transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        current_time_offset: float,
    ) -> List[Dict]:

        try:
            # ── Fix 1: validate file before sending to Whisper ────────────────
            import os
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:
                logger.warning(
                    "Whisper: file too small (%d bytes) — skipping: %s",
                    file_size, audio_path,
                )
                return []

            transcribe_kwargs = {}

            if language:
                transcribe_kwargs["language"] = language

            segments, info = self.model.transcribe(
                audio_path,
                **transcribe_kwargs,
            )

            # ── Fix 2: check language probability ─────────────────────────────
            # Low probability = audio is silent or garbage
            if info.language_probability < 0.4:
                logger.warning(
                    "Whisper: low language confidence %.2f for %s — skipping",
                    info.language_probability, audio_path,
                )
                return []

            print(
                f"🎤 Whisper detected: lang={info.language} "
                f"prob={info.language_probability:.2f}"
            )

            subtitles = []

            for seg in segments:
                text = seg.text.strip()

                if not text:
                    continue

                subtitles.append(
                    {
                        "text": text,
                        "timestamp": round(seg.start + current_time_offset, 2),
                        "duration": round(seg.end - seg.start, 2),
                        "source": "whisper",
                        "language": info.language,
                    }
                )

            print(
                f"✅ Whisper: {len(subtitles)} segments "
                f"| lang={info.language} "
                f"| prob={info.language_probability:.2f}"
            )

            return subtitles

        # ── Fix 3: catch the exact crash you saw ──────────────────────────────
        except ValueError as e:
            if "empty sequence" in str(e):
                logger.warning(
                    "Whisper: empty audio sequence at offset=%.1fs — "
                    "video likely ended. Skipping: %s",
                    current_time_offset, audio_path,
                )
            else:
                logger.error("Whisper ValueError: %s", e)
            return []

        except Exception as e:
            print(f"⚠️ Whisper sync error: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ── YouTube Segment ────────────────────────────────────────────────────

    async def transcribe_youtube_segment(
        self,
        video_id: str,
        start_time: float,
        duration: float = 60.0,
        language: Optional[str] = None,
    ) -> List[Dict]:

        if not self.enabled:
            return []

        from .audio_extractor import audio_extractor
        from .audio_trimmer import audio_trimmer

        print(
            f"🎤 Whisper: Processing {video_id} "
            f"@ {start_time:.1f}s "
            f"(duration={duration:.0f}s)"
        )

        # -------------------------------------------------------
        # Step 1: Download (or reuse cached) full audio
        # -------------------------------------------------------

        full_audio = await audio_extractor.extract_audio_segment(
            video_id=video_id
        )

        if not full_audio:
            print("⚠️ Unable to get cached audio")
            return []

        # -------------------------------------------------------
        # Step 2: Trim required segment using FFmpeg
        # -------------------------------------------------------

        clip = await audio_trimmer.trim_audio(
            input_file=full_audio,
            start_time=start_time,
            duration=duration,
        )

        if not clip:
            print("⚠️ FFmpeg trimming failed")
            return []

        try:

            print("🎤 Sending clip to Whisper...")

            subtitles = await self.transcribe_file(
                audio_path=clip,
                language=language,
                current_time_offset=start_time,
            )

            return subtitles

        finally:

            audio_trimmer.cleanup(clip)


# Global instance
whisper_service = WhisperService()