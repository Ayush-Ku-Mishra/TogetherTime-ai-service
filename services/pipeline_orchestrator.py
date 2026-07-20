"""
Pipeline Orchestrator

Coordinates the complete AI ingestion pipeline.

                         TogetherTime AI Backend

                                 │
                                 ▼
                      User pastes YouTube URL
                                 │
                                 ▼
                    Already processed? (Step 3)
                                 │
                    YES ─────────┘─────────► Return immediately
                                 │
                    NO ──────────┘
                                 │
                                 ▼
                         Audio Extractor
                      (Download / Cache Audio)
                                 │
                                 ▼
                      Processing Queue (Phase 1.2)
                   (Decides which window comes next)
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Window Loop           │
                    │                        │
                    │  while queue not empty │
                    │    Audio Trimmer       │
                    │    Whisper Service     │
                    │    Subtitle Chunker    │
                    │    Scene Memory        │
                    │    mark_processed()    │
                    └────────────────────────┘
                                 │
                                 ▼
                    Mark video as processed (Step 3)
                                 │
                                 ▼
                           Scene Objects
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  Rolling Memory             Qdrant                  Future Timeline
 (Redis / Cache)        (Semantic Memory)          (Frontend Scenes)
        │                        │
        └──────────────┬─────────┘
                       ▼
                Context Builder
                       │
                       ▼
                  LLM Service
                       │
                       ▼
                  Final AI Answer

Responsibilities
----------------
✅ Coordinate services
✅ Skip already-processed videos (Step 3)
✅ Process complete video window-by-window
✅ Accept subtitle push from frontend
✅ Manage caption store per room
✅ Track processing progress per room
✅ Report health of all services

Does NOT
---------
❌ Decide which window comes next  — ProcessingQueue owns that
❌ Summarize scenes                — SceneMemory owns that
❌ Know about Redis/Qdrant         — SceneMemory owns that
❌ Build prompts                   — ContextBuilder owns that
❌ Call Gemini                     — LLMService owns that
❌ Answer questions                — AIRouter owns that
❌ Trim audio                      — AudioTrimmer owns that
❌ Transcribe audio                — WhisperService owns that
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .audio_extractor  import audio_extractor
from .audio_trimmer    import audio_trimmer
from .whisper_service  import whisper_service
from .subtitle_chunker import subtitle_chunker
from .scene_memory     import scene_memory
from .processing_queue import processing_queue
from .video_metadata   import video_metadata
import asyncio
from .realtime_memory  import realtime_memory
from .state_persistence import state_persistence
from asyncio import Semaphore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level stores
# ─────────────────────────────────────────────────────────────────────────────

# Shape: { room_id: { video_id: [ caption_dict, ... ] } }
_caption_store: Dict[str, Dict[str, List[Dict]]] = {}

# Step 3.1 — track which video each room has fully processed
# Shape: { room_id: video_id }
_processed_videos: Dict[str, str] = {}
_processing_locks: Dict[str, asyncio.Lock] = {}


def get_captions(
    room_id:           str,
    video_id:          str,
    current_timestamp: float,
    window_seconds:    float = 60.0,
) -> List[Dict]:
    """
    Return captions for a room/video that fall within
    [current_timestamp - window_seconds, current_timestamp + window_seconds].
    """
    vid_caps = _caption_store.get(room_id, {}).get(video_id, [])

    if not vid_caps:
        return []

    lo = max(0.0, current_timestamp - window_seconds)
    hi = current_timestamp + window_seconds

    return [c for c in vid_caps if lo <= c.get("timestamp", 0) <= hi]


# ─────────────────────────────────────────────────────────────────────────────
# PipelineOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    Thin coordinator — knows the order of steps, not their internals.

    Two public entry points
    -----------------------
    process_video()          Full video ingestion, window-by-window
    process_subtitle_push()  Real-time subtitle push from the frontend
    """

    # =========================================================================
    # INIT
    # =========================================================================

    def __init__(self):
        # { room_id: { processing: bool, progress: int } }
        self.processing_status: Dict[str, Dict] = {}
        self._whisper_semaphore = Semaphore(2)
        logger.info("✅ Pipeline Orchestrator initialized")

    # =========================================================================
    # PUBLIC — PROCESSING STATE
    # =========================================================================

    def start_processing(self, room_id: str) -> None:
        self.processing_status[room_id] = {
            "processing": True,
            "progress":   0,
        }
        logger.info("▶ Processing started — room=%s", room_id)

    def update_progress(self, room_id: str, progress: int) -> None:
        if room_id in self.processing_status:
            self.processing_status[room_id]["progress"] = progress
            logger.info("⏳ Progress %d%% — room=%s", progress, room_id)

    def finish_processing(self, room_id: str) -> None:
        self.processing_status[room_id] = {
            "processing": False,
            "progress":   100,
        }
        logger.info("✅ Processing finished — room=%s", room_id)

    def is_processing(self, room_id: str) -> bool:
        return self.processing_status.get(
            room_id, {}
        ).get("processing", False)

    def get_progress(self, room_id: str) -> int:
        return self.processing_status.get(
            room_id, {}
        ).get("progress", 0)

    # =========================================================================
    # PUBLIC — FULL VIDEO PIPELINE
    # =========================================================================

    async def process_video(
        self,
        *,
        room_id:          str,
        video_id:         str,
        movie_title:      str           = "",
        language:         Optional[str] = None,
        video_duration_s: float         = 0.0,
        platform:         str           = "youtube",
    ) -> Dict:

        lock_key = f"{room_id}:{video_id}"

        if lock_key not in _processing_locks:
            _processing_locks[lock_key] = asyncio.Lock()

        if _processing_locks[lock_key].locked():
            logger.info(
                "process_video: already running — room=%s video=%s",
                room_id, video_id,
            )
            return {
                "success":  True,
                "cached":   True,
                "reason":   "already_processing",
                "room_id":  room_id,
                "video_id": video_id,
            }

        async with _processing_locks[lock_key]:

            t0 = time.perf_counter()

            # ── Check if fully processed ───────────────────────────────────────
            # Check Redis first (survives restart)
            saved_video = state_persistence.load_processed_video(room_id)

            if saved_video == video_id:
                logger.info(
                    "✅ Already fully processed — room=%s video=%s",
                    room_id, video_id,
                )
                _processed_videos[room_id] = video_id
                # Still restore realtime for answering questions
                realtime_memory.restore_from_redis(room_id, video_id)
                self.finish_processing(room_id)
                return {
                    "success":  True,
                    "cached":   True,
                    "room_id":  room_id,
                    "video_id": video_id,
                }

            if _processed_videos.get(room_id) == video_id:
                logger.info(
                    "✅ Already processed (memory) — room=%s video=%s",
                    room_id, video_id,
                )
                return {
                    "success":  True,
                    "cached":   True,
                    "room_id":  room_id,
                    "video_id": video_id,
                }

            # ── Restore realtime transcripts ───────────────────────────────────
            realtime_memory.restore_from_redis(room_id, video_id)

            # ── Fetch metadata ─────────────────────────────────────────────────
            metadata = await video_metadata.get_metadata(
                platform=platform,
                video_id=video_id,
            )

            print("=" * 60)
            print("VIDEO METADATA")
            print(f"Title     : {metadata['title']}")
            print(f"Channel   : {metadata['channel']}")
            print(f"Duration  : {metadata['duration']}")
            print(f"Published : {metadata['publishedAt']}")
            print("=" * 60)

            result: Dict = {
                "success":         False,
                "cached":          False,
                "room_id":         room_id,
                "video_id":        video_id,
                "movie_title":     movie_title or metadata["title"],
                "total_windows":   0,
                "total_subtitles": 0,
                "total_chunks":    0,
                "total_scenes":    0,
                "failed_windows":  0,
                "processing_time": 0.0,
            }

            self.start_processing(room_id)

            try:
                # ── Download / use cached audio ────────────────────────────────
                audio_path = await self._extract_audio(
                    video_id=video_id,
                    room_id=room_id,
                )

                if not audio_path:
                    logger.error("Audio extraction failed")
                    self.finish_processing(room_id)
                    return result

                self.update_progress(room_id, 5)

                # ── Get duration ───────────────────────────────────────────────
                duration_s = (
                    video_duration_s
                    or audio_extractor.get_duration(audio_path)
                    or 7200.0
                )

                print("=" * 60)
                print("ACTUAL AUDIO DURATION:", duration_s)
                print("=" * 60)

                # ── Register with queue ────────────────────────────────────────
                # start_video restores from Redis if available
                current_playback = processing_queue.get_current_playback(
                    room_id=room_id,
                    video_id=video_id,
                )

                start_from = max(0.0, current_playback - 30.0)

                processing_queue.start_video(
                    room_id=room_id,
                    video_id=video_id,
                    total_duration_s=duration_s,
                    start_from_s=start_from,
                )

                restored_until = processing_queue.get_processed_until(
                    room_id=room_id,
                    video_id=video_id,
                )

                logger.info(
                    "Starting window loop from %.1fs",
                    restored_until,
                )

                # ── Window loop ────────────────────────────────────────────────
                windows_done = 0

                while True:

                    if processing_queue.should_pause(room_id, video_id):
                        logger.info("⏸ Paused — room=%s", room_id)
                        while processing_queue.should_pause(room_id, video_id):
                            await asyncio.sleep(5.0)
                        logger.info("▶ Resumed — room=%s", room_id)

                    processing_queue.refill_queue(room_id, video_id)
                    window = processing_queue.get_next_window(
                        room_id, video_id
                    )

                    print("=" * 60)
                    print("WINDOW:", window)
                    print("=" * 60)

                    if window is None:
                        logger.info("Queue empty — done")
                        break

                    if window["start"] >= duration_s - 0.5:
                        processing_queue.mark_processed(
                            room_id, video_id, duration_s
                        )
                        break

                    windows_done += 1

                    win_result = await self._process_window(
                        room_id=room_id,
                        video_id=video_id,
                        audio_path=audio_path,
                        win_start=window["start"],
                        win_dur=window["duration"],
                        language=language,
                        win_index=windows_done,
                    )

                    window_end = window["start"] + window["duration"]

                    jumped = processing_queue.consume_jump(room_id, video_id)

                    if jumped:
                        processing_queue.refill_queue(room_id, video_id)
                    else:
                        processing_queue.mark_processed(
                            room_id, video_id, window_end
                        )
                        processing_queue.refill_queue(room_id, video_id)

                    result["total_subtitles"] += win_result["subtitles"]
                    result["total_chunks"]    += win_result["chunks"]
                    result["total_scenes"]    += win_result["scenes"]

                    if not win_result["success"]:
                        result["failed_windows"] += 1

                    q_status = processing_queue.status(room_id, video_id)
                    pct      = q_status.get("pct_processed", 0)
                    self.update_progress(room_id, 5 + int(pct * 0.90))

                result["total_windows"]   = windows_done
                result["success"]         = True
                result["processing_time"] = round(
                    time.perf_counter() - t0, 2
                )

                # Mark as fully processed
                _processed_videos[room_id] = video_id
                state_persistence.save_processed_video(room_id, video_id)

                self.finish_processing(room_id)
                return result

            except Exception:
                logger.exception("process_video failed")
                self.finish_processing(room_id)
                return result


    # =========================================================================
    # PUBLIC — REAL-TIME SUBTITLE PUSH
    # =========================================================================

    async def process_subtitle_push(
        self,
        *,
        room_id:       str,
        video_id:      str,
        timestamp:     float,
        platform:      str        = "youtube",
        movie_title:   str        = "",
        frontend_subs: List[Dict] = [],
    ) -> Dict:
        """
        Accept a subtitle batch from the frontend.
        """

        result: Dict = {
            "processed":       False,
            "subtitles_used":  0,
            "subtitle_source": "none",
            "content_type":    "unknown",
            "summary_created": False,
            "summary":         "",
            "tmdb_data":       None,
        }

        try:
            if not frontend_subs:
                return result

            self._store_captions(room_id, video_id, frontend_subs)

            result["subtitles_used"]  = len(frontend_subs)
            result["subtitle_source"] = "frontend"
            result["processed"]       = True
            result["content_type"]    = "video"

            chunks = subtitle_chunker.build_chunks(frontend_subs)

            if not chunks:
                return result

            scene_result = await scene_memory.process_chunks(
                room_id=room_id,
                video_id=video_id,
                chunks=chunks,
                source="frontend",
            )

            result["summary_created"] = scene_result.get("stored", False)
            result["summary"]         = scene_result.get("summary", "")

            return result

        except Exception:
            logger.exception("process_subtitle_push failed")
            return result

    # =========================================================================
    # PUBLIC — CLEAR ROOM
    # =========================================================================

    def clear_room(self, room_id: str) -> None:
        """
        Remove all in-process state for a room.
        """
        _caption_store.pop(room_id, None)
        self.processing_status.pop(room_id, None)
        processing_queue.clear_room(room_id)

        # Step 3.4 — reset processed video so a new session starts fresh
        _processed_videos.pop(room_id, None)
        from .realtime_memory import realtime_memory
        realtime_memory.clear(room_id)
        logger.info("🗑 Cleared room state — room=%s", room_id)

    # =========================================================================
    # PUBLIC — HEALTH
    # =========================================================================

    async def health(self) -> Dict:
        audio_ok      = audio_extractor.enabled
        whisper_ok    = whisper_service.enabled
        summarizer_ok = await scene_memory.summarizer_healthy()
        redis_ok      = await scene_memory.redis_healthy()
        qdrant_ok     = await scene_memory.qdrant_healthy()

        report = {
            "audio":      audio_ok,
            "whisper":    whisper_ok,
            "summarizer": summarizer_ok,
            "redis":      redis_ok,
            "qdrant":     qdrant_ok,
        }
        report["healthy"] = all(report.values())

        logger.info("Health report: %s", report)
        return report

    # =========================================================================
    # PRIVATE — SINGLE WINDOW
    # =========================================================================

    async def _process_window(
        self,
        *,
        room_id:    str,
        video_id:   str,
        audio_path: str,
        win_start:  float,
        win_dur:    float,
        language:   Optional[str],
        win_index:  int,
    ) -> Dict:

        win_result = {
            "success":   False,
            "subtitles": 0,
            "chunks":    0,
            "scenes":    0,
        }

        logger.info(
            "  Window %d  %.1f → %.1fs",
            win_index, win_start, win_start + win_dur,
        )

        clip_path = await self._trim_window(audio_path, win_start, win_dur)

        if not clip_path:
            logger.warning("  Window %d — trim failed, skipping", win_index)
            return win_result

        subtitles = await self._transcribe_window(
            clip_path, language, win_start
        )

        audio_trimmer.cleanup(clip_path)

        if not subtitles:
            logger.warning("  Window %d — no subtitles, skipping", win_index)
            return win_result

        win_result["subtitles"] = len(subtitles)

        # ── Store transcript to RealtimeMemory IMMEDIATELY ────────────────────
        # This happens BEFORE Gemini — transcript available within seconds
        transcript_text = " ".join(
            s.get("text", "") for s in subtitles
            if s.get("text", "").strip()
        )

        if transcript_text:
            realtime_memory.store_transcript(
                room_id=room_id,
                video_id=video_id,
                start=win_start,
                end=win_start + win_dur,
                transcript=transcript_text,
                language=subtitles[0].get("language", "") if subtitles else "",
            )
            logger.info(
                "  ✅ Transcript stored immediately — window %d  %.1f→%.1fs",
                win_index, win_start, win_start + win_dur,
            )

        # ── Mark transcript_ready IMMEDIATELY ─────────────────────────────────
        # processed_until now advances here — not after Gemini finishes
        win_result["success"] = True
        win_result["chunks"]  = 1   # at minimum we have transcript
        win_result["scenes"]  = 0   # scenes come later from background

        # ── Run Gemini summarization in BACKGROUND ────────────────────────────
        # Don't await this — fire and forget
        # Queue moves to next window immediately
        chunks = subtitle_chunker.build_chunks(subtitles)

        if chunks:
            win_result["chunks"] = len(chunks)
            asyncio.create_task(
                self._background_summarize(
                    room_id=room_id,
                    video_id=video_id,
                    chunks=chunks,
                    win_index=win_index,
                )
            )
            logger.info(
                "  🔄 Background summarization started — window %d",
                win_index,
            )

        return win_result

    async def _process_window_with_semaphore(
        self,
        *,
        room_id:    str,
        video_id:   str,
        audio_path: str,
        win_start:  float,
        win_dur:    float,
        language:   Optional[str],
        win_index:  int,
    ) -> tuple[Dict, Dict]:
        """Wraps _process_window with semaphore for parallel limiting."""

        async with self._whisper_semaphore:
            result = await self._process_window(
                room_id=room_id,
                video_id=video_id,
                audio_path=audio_path,
                win_start=win_start,
                win_dur=win_dur,
                language=language,
                win_index=win_index,
            )
            return {"start": win_start, "duration": win_dur}, result

    async def _background_summarize(
        self,
        room_id:   str,
        video_id:  str,
        chunks:    List[Dict],
        win_index: int,
    ) -> None:
        """
        Run Gemini summarization + Qdrant storage in the background.

        This runs AFTER the window loop has already moved to the next window.
        Never blocks Whisper from starting the next window.
        """
        try:
            scene_result = await scene_memory.process_chunks(
                room_id=room_id,
                video_id=video_id,
                chunks=chunks,
                source="whisper",
            )

            logger.info(
                "  ✅ Background summarization done — window %d  scenes=%d",
                win_index, scene_result.get("stored", 0),
            )

        except Exception:
            logger.exception(
                "  ❌ Background summarization failed — window %d", win_index
            )

    # =========================================================================
    # PRIVATE — HELPERS
    # =========================================================================

    async def _extract_audio(
        self,
        video_id: str,
        room_id:  str = "",
    ) -> Optional[str]:
        """
        Get audio file for a video.
        
        Uses cached file if available (no re-download).
        room_id used for cache folder organization.
        """
        try:
            path = await audio_extractor.extract_audio_segment(
                video_id=video_id,
                room_id=room_id,
            )
            if path:
                logger.info("Audio ready: %s", path)
            else:
                logger.error(
                    "_extract_audio: no path returned — "
                    "room=%s video=%s",
                    room_id, video_id,
                )
            return path
        except Exception:
            logger.exception(
                "_extract_audio error — room=%s video=%s",
                room_id, video_id,
            )
            return None

    async def _trim_window(
        self,
        audio_path: str,
        start:      float,
        duration:   float,
    ) -> Optional[str]:
        try:
            return await audio_trimmer.trim_audio(
                input_file=audio_path,
                start_time=start,
                duration=duration,
            )
        except Exception:
            logger.exception("_trim_window error  start=%.1f", start)
            return None

    async def _transcribe_window(
        self,
        clip_path:   str,
        language:    Optional[str],
        time_offset: float,
    ) -> List[Dict]:
        try:
            return await whisper_service.transcribe_file(
                audio_path=clip_path,
                language=language,
                current_time_offset=time_offset,
            )
        except Exception:
            logger.exception(
                "_transcribe_window error  clip=%s",
                clip_path,
            )
            return []

    def _store_captions(
        self,
        room_id:  str,
        video_id: str,
        captions: List[Dict],
    ) -> None:
        room = _caption_store.setdefault(room_id, {})
        buf  = room.setdefault(video_id, [])
        buf.extend(captions)
        if len(buf) > 500:
            room[video_id] = buf[-500:]


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

pipeline_orchestrator = PipelineOrchestrator()