"""
Processing Queue

Decides which audio window should be processed next.

STEP 4 — Priority-based processing
----------------------------------
Always stay ahead of current playback.

If user is at 2:30 and we've only processed up to 1:30,
we immediately jump to 2:30 and process forward from there,
instead of continuing sequentially from 1:30.

State shape (per room/video)
----------------------------
{
    "video_id":          str,
    "total_duration_s":  float,
    "current_time":      float,   # 🆕 Step 4: where user is NOW
    "processed_until_s": float,
    "queue": [...],
    "processing":        bool,    # 🆕 Step 4: prevent race conditions
}

Rules
-----
- Windows are 30 s wide with 2 s overlap
- We always stay LOOKAHEAD_SECONDS ahead of current_time
- When current_time jumps forward, we re-prioritize the queue
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from .state_persistence import state_persistence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SECONDS:    float = 30.0
WINDOW_OVERLAP:    float = 2.0
LOOKAHEAD_SECONDS: float = 120.0   # stay 2 minutes ahead
PAUSE_THRESHOLD:   float = 150.0 
RESUME_THRESHOLD:  float = 60.0 


# ─────────────────────────────────────────────────────────────────────────────
# ProcessingQueue
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingQueue:
    """
    Step 4: Priority queue that always processes windows ahead of playback first.
    """

    def __init__(self):
        # { (room_id, video_id): state_dict }
        self._state: Dict[tuple, Dict] = {}
        logger.info("✅ ProcessingQueue initialized (Step 4)")

    # =========================================================================
    # PUBLIC — START
    # =========================================================================

    def start_video(
        self,
        room_id:          str,
        video_id:         str,
        total_duration_s: float,
        start_from_s:     float = 0.0,
    ) -> None:

        key = (room_id, video_id)

        # ── Already in memory — just refill, don't reset ───────────────────────
        if key in self._state:
            existing = self._state[key]
            existing["total_duration_s"] = total_duration_s

            logger.info(
                "start_video: already in memory — "
                "resuming from %.1fs  room=%s video=%s",
                existing["processed_until_s"],
                room_id, video_id,
            )

            # Just refill the queue from current position
            self._refill_queue(key)

            from .state_persistence import state_persistence
            state_persistence.register_room(room_id)
            return

        # ── Try restore from Redis ─────────────────────────────────────────────
        from .state_persistence import state_persistence
        saved = state_persistence.load_queue_state(room_id, video_id)

        if saved:
            restored_processed = saved.get("processed_until_s", 0.0)
            restored_current   = saved.get("current_time",      0.0)

            logger.info(
                "start_video: RESTORED from Redis — "
                "room=%s video=%s processed=%.1fs ✅",
                room_id, video_id, restored_processed,
            )

            self._state[key] = {
                "video_id":          video_id,
                "total_duration_s":  total_duration_s,
                "current_time":      restored_current,
                "processed_until_s": restored_processed,
                "queue":             [],
                "processing":        False,
                "jump_requested":    False,
                "jump_target":       0.0,
            }

        else:
            # ── Fresh start ────────────────────────────────────────────────────
            logger.info(
                "start_video: FRESH — room=%s video=%s",
                room_id, video_id,
            )

            self._state[key] = {
                "video_id":          video_id,
                "total_duration_s":  total_duration_s,
                "current_time":      start_from_s,
                "processed_until_s": start_from_s,
                "queue":             [],
                "processing":        False,
                "jump_requested":    False,
                "jump_target":       0.0,
            }

        state_persistence.register_room(room_id)
        self._refill_queue(key)

        logger.info(
            "start_video — room=%s video=%s "
            "duration=%.1fs start=%.1fs queue=%d",
            room_id, video_id, total_duration_s,
            self._state[key]["processed_until_s"],
            len(self._state[key]["queue"]),
        )


    # =========================================================================
    # PUBLIC — PLAYBACK POSITION  (🆕 Step 4)
    # =========================================================================

    def update_playback(
        self,
        room_id:      str,
        video_id:     str,
        current_time: float,
    ) -> None:

        key = (room_id, video_id)

        # Simple — if not in memory, just log and return
        # Frontend will call /process-video to restart
        if key not in self._state:
            logger.warning(
                "update_playback: unknown room/video — "
                "room=%s video=%s",
                room_id, video_id,
            )
            return

        state   = self._state[key]
        old_pos = state["current_time"]
        state["current_time"] = current_time

        # Persist to Redis
        from .state_persistence import state_persistence
        state_persistence.save_queue_state(room_id, video_id, state)

        if current_time > state["processed_until_s"] + 60.0:
            logger.info(
                "⚡ JUMP DETECTED — room=%s  "
                "current=%.1fs  processed=%.1fs",
                room_id, current_time,
                state["processed_until_s"],
            )
            state["jump_requested"] = True
            state["jump_target"]    = current_time
            self._reprioritize_queue(key)
        else:
            self._refill_queue(key)

    def _try_restore(
        self,
        room_id:  str,
        video_id: str,
    ) -> bool:
        """
        Restore queue state from Redis synchronously.
        Only restores queue — NOT realtime memory.
        Returns True if restored.
        """

        from .state_persistence import state_persistence

        saved = state_persistence.load_queue_state(room_id, video_id)

        if not saved:
            return False

        key = (room_id, video_id)

        self._state[key] = {
            "video_id":          video_id,
            "total_duration_s":  saved.get("total_duration_s",  0.0),
            "current_time":      saved.get("current_time",      0.0),
            "processed_until_s": saved.get("processed_until_s", 0.0),
            "queue":             [],
            "processing":        False,
            "jump_requested":    False,
            "jump_target":       0.0,
        }

        self._refill_queue(key)

        logger.info(
            "_try_restore: queue restored — room=%s video=%s "
            "processed=%.1fs ✅",
            room_id, video_id,
            self._state[key]["processed_until_s"],
        )

        return True


    def consume_jump(
        self,
        room_id:  str,
        video_id: str,
    ) -> bool:
        """
        🆕 Called by the processing loop after each window.

        Returns True if a jump was requested (and resets the flag).
        The loop should skip mark_processed for the old window
        and restart from the new position.
        """

        key = (room_id, video_id)

        if key not in self._state:
            return False

        state = self._state[key]

        if state.get("jump_requested", False):
            logger.info(
                "consume_jump — room=%s  jumping to %.1fs",
                room_id, state["jump_target"],
            )
            state["jump_requested"]    = False
            state["processed_until_s"] = max(
                state["processed_until_s"],
                state["jump_target"] - 60.0,  # start 60s before target
            )
            return True

        return False

    def get_jump_target(
        self,
        room_id:  str,
        video_id: str,
    ) -> float:
        """Return where the user jumped to."""

        key = (room_id, video_id)

        if key not in self._state:
            return 0.0

        return self._state[key].get("jump_target", 0.0)

        
    def should_pause(
        self,
        room_id:  str,
        video_id: str,
    ) -> bool:
        """
        Return True if we're far enough ahead that processing should pause.

        Prevents racing through an entire 2hr movie when user is at minute 10.
        Processing resumes automatically when user catches up.
        """
        key = (room_id, video_id)

        if key not in self._state:
            return False

        state       = self._state[key]
        buffer_size = state["processed_until_s"] - state["current_time"]

        should = buffer_size >= PAUSE_THRESHOLD

        if should:
            logger.info(
                "should_pause=True — buffer=%.1fs >= threshold=%.1fs  room=%s",
                buffer_size, PAUSE_THRESHOLD, room_id,
            )

        return should

    def should_resume(
        self,
        room_id:  str,
        video_id: str,
    ) -> bool:
        """
        Return True if buffer has shrunk enough that processing should resume.
        """
        key = (room_id, video_id)

        if key not in self._state:
            return False

        state       = self._state[key]
        buffer_size = state["processed_until_s"] - state["current_time"]

        return buffer_size < RESUME_THRESHOLD

    # =========================================================================
    # PUBLIC — GET NEXT WINDOW
    # =========================================================================

    def get_next_window(
        self,
        room_id:  str,
        video_id: str,
    ) -> Optional[Dict]:
        """
        Pop and return the highest-priority window.

        Priority = closest to current_time that hasn't been processed yet.
        """

        key = (room_id, video_id)

        if key not in self._state:
            return None

        state = self._state[key]

        if not state["queue"]:
            return None

        # Sort queue by start time (ascending) so we always get
        # the window closest to current_time first
        state["queue"].sort(key=lambda w: w["start"])

        window = state["queue"].pop(0)
        state["processing"] = True    # 🆕 Step 4: mark as busy

        logger.info(
            "get_next_window — room=%s  %.1f → %.1fs  "
            "(queue remaining: %d, current_time: %.1fs)",
            room_id,
            window["start"],
            window["start"] + window["duration"],
            len(state["queue"]),
            state["current_time"],
        )

        return window

    # =========================================================================
    # PUBLIC — MARK PROCESSED  (🆕 Step 4: accepts end_time directly)
    # =========================================================================

    def mark_processed(
        self,
        room_id:  str,
        video_id: str,
        end_time: float,
    ) -> None:

        key = (room_id, video_id)

        if key not in self._state:
            return

        state = self._state[key]

        if end_time > state["processed_until_s"]:
            state["processed_until_s"] = end_time

        state["processing"] = False

        # ── Persist to Redis immediately ───────────────────────────────────────
        state_persistence.save_queue_state(room_id, video_id, state)

        if state["processed_until_s"] >= state["total_duration_s"] - 0.5:
            state["queue"].clear()
            logger.info(
                "mark_processed: video fully processed — "
                "clearing queue  room=%s  processed=%.3fs  total=%.3fs",
                room_id,
                state["processed_until_s"],
                state["total_duration_s"],
            )
            return

        self._refill_queue(key)

        next_window = (
            f'{state["queue"][0]["start"]:.1f} → '
            f'{state["queue"][0]["start"] + state["queue"][0]["duration"]:.1f}'
            if state["queue"]
            else "None"
        )

        logger.info(
            "\n"
            "========================================================\n"
            "PROCESSING QUEUE STATUS\n\n"
            "Room         : %s\n"
            "Video        : %s\n\n"
            "🎬 Playback  : %.1fs\n"
            "📦 Processed : %.1fs\n\n"
            "📋 Queue     : %d window(s)\n"
            "➡ Next       : %s\n\n"
            "Ready        : %s\n"
            "Processing   : %s\n"
            "========================================================",
            room_id,
            state["video_id"],
            state["current_time"],
            state["processed_until_s"],
            len(state["queue"]),
            next_window,
            "YES" if self.is_ready(room_id, video_id) else "NO",
            "YES" if state["processing"] else "NO",
        )

    # =========================================================================
    # PUBLIC — GET PROCESSED UNTIL  (🆕 Step 4)
    # =========================================================================

    def get_processed_until(
        self,
        room_id:  str,
        video_id: str,
    ) -> float:
        """
        Return how far we've processed in seconds.

        Used by /playback-update endpoint to tell frontend
        how far ahead the AI is.
        """

        key = (room_id, video_id)

        if key not in self._state:
            return 0.0

        return self._state[key]["processed_until_s"]

    # =========================================================================
    # PUBLIC — GET CURRENT PLAYBACK  (needed by VideoContextService)
    # =========================================================================

    def get_current_playback(
        self,
        room_id:  str,
        video_id: str,
    ) -> float:
        """
        Return the last known playback position for this room/video.

        Updated every time the frontend sends a /playback-update request.
        Returns 0.0 if the room/video is unknown.

        Parameters
        ----------
        room_id  : watch party room identifier
        video_id : YouTube video id

        Returns
        -------
        Playback position in seconds, or 0.0.
        """

        key = (room_id, video_id)

        if key not in self._state:
            return 0.0

        return self._state[key].get("current_time", 0.0)


    def get_current_playback(
        self,
        room_id: str,
        video_id: str,
    ) -> float:
        key = (room_id, video_id)

        if key not in self._state:
            return 0.0

        return self._state[key]["current_time"]

    # =========================================================================
    # PUBLIC — IS READY
    # =========================================================================

    def is_ready(
        self,
        room_id:  str,
        video_id: str,
    ) -> bool:
        """
        Return True if we're far enough ahead of playback.
        """

        key = (room_id, video_id)

        if key not in self._state:
            return False

        state = self._state[key]

        fully_done = state["processed_until_s"] >= state["total_duration_s"]
        far_enough = (
            state["processed_until_s"]
            >= state["current_time"] + LOOKAHEAD_SECONDS
        )

        return fully_done or far_enough

    # =========================================================================
    # PUBLIC — CLEAR ROOM
    # =========================================================================

    def clear_room(self, room_id: str) -> None:
        """Remove all queue state for every video in a room."""

        keys_to_remove = [k for k in self._state if k[0] == room_id]

        for key in keys_to_remove:
            del self._state[key]

        logger.info(
            "clear_room — room=%s  removed %d queue(s)",
            room_id, len(keys_to_remove),
        )

    # =========================================================================
    # PUBLIC — STATUS
    # =========================================================================

    def status(
        self,
        room_id:  str,
        video_id: str,
    ) -> Dict:
        """Return a human-readable snapshot of the queue state."""

        key = (room_id, video_id)

        if key not in self._state:
            return {"exists": False}

        state = self._state[key]

        total = state["total_duration_s"]
        until = state["processed_until_s"]
        pct   = round((until / total * 100) if total > 0 else 0.0, 1)
        queue = state["queue"]

        return {
            "exists":            True,
            "video_id":          state["video_id"],
            "total_duration_s":  total,
            "current_time":      state["current_time"],      # 🆕 Step 4
            "processed_until_s": until,
            "queue_length":      len(queue),
            "next_window_start": queue[0]["start"] if queue else None,
            "is_ready":          self.is_ready(room_id, video_id),
            "processing":        state["processing"],        # 🆕 Step 4
            "pct_processed":     pct,
        }

        # =========================================================================
    # PUBLIC — DEBUG STATUS
    # =========================================================================

    def get_debug_status(
        self,
        room_id: str,
        video_id: str,
    ) -> dict:
        """
        Return the current processing state for debugging.
        """

        return self.status(room_id, video_id)

    # =========================================================================
    # PRIVATE — REFILL QUEUE
    # =========================================================================

    def _refill_queue(self, key: tuple) -> None:
        """
        Add windows to the queue until we reach the lookahead target.

        Fix: Use 0.5s threshold instead of 0 to handle floating point drift.
            Never generate a window whose start >= total_duration.
        """

        state = self._state[key]

        total  = state["total_duration_s"]
        target = min(
            state["current_time"] + LOOKAHEAD_SECONDS,
            total,
        )

        # Where do we generate from?
        if state["queue"]:
            last     = state["queue"][-1]
            gen_from = last["start"] + last["duration"]
        else:
            gen_from = max(
                state["processed_until_s"],
                state["current_time"],
            )

        step  = WINDOW_SECONDS - WINDOW_OVERLAP
        added = 0

        while gen_from < target:

            remaining = total - gen_from

            # ── Fix 1: use threshold not zero ─────────────────────────────────
            # Floating point: 148.0 - 148.0 might be 1e-10, not exactly 0.
            # Use 0.5s threshold so we never generate a meaningless window.
            if remaining < 0.5:
                logger.debug(
                    "_refill_queue: remaining=%.6f < 0.5s threshold, "
                    "stopping at %.3fs (total=%.3fs)",
                    remaining, gen_from, total,
                )
                break

            # ── Fix 2: never start a window at or past total duration ──────────
            if gen_from >= total:
                logger.debug(
                    "_refill_queue: gen_from=%.3fs >= total=%.3fs, stopping",
                    gen_from, total,
                )
                break

            duration = min(WINDOW_SECONDS, remaining)

            # ── Fix 3: never add a sub-second window ──────────────────────────
            if duration < 1.0:
                logger.debug(
                    "_refill_queue: duration=%.6f < 1.0s, stopping at %.3fs",
                    duration, gen_from,
                )
                break

            # ── Only add if not already processed ─────────────────────────────
            if gen_from >= state["processed_until_s"] - WINDOW_OVERLAP:
                state["queue"].append({
                    "start":    round(gen_from, 2),
                    "duration": round(duration, 2),
                })
                added += 1

            gen_from = round(gen_from + step, 2)

        if added:
            logger.debug(
                "_refill_queue — key=%s  added=%d  queue=%d",
                key, added, len(state["queue"]),
            )

    # =========================================================================
    # PRIVATE — RE-PRIORITIZE QUEUE  (🆕 Step 4)
    # =========================================================================

    def _reprioritize_queue(self, key: tuple) -> None:
        """
        Clear the old queue and rebuild from current_time forward.

        Called when user seeks forward in the video —
        we don't want to waste time processing windows they skipped.
        """

        state = self._state[key]

        # Clear old queue
        state["queue"].clear()

        # Rebuild from current playback position
        logger.info(
            "_reprioritize_queue — key=%s  "
            "clearing old queue, rebuilding from %.1fs",
            key, state["current_time"],
        )

        self._refill_queue(key)

    # =========================================================================
    # PUBLIC — GET NEXT WINDOW  (Updated for Step 5)
    # =========================================================================

    def get_next_window(
        self,
        room_id:  str,
        video_id: str = "",
    ) -> Optional[Dict]:
        """
        Pop and return the highest-priority window.

        Step 5: Simplified — just pop from queue.
        The queue is already sorted by priority in _refill_queue().

        Returns
        -------
        {"start": float, "duration": float}  or  None
        """

        # Try with video_id first
        key = (room_id, video_id) if video_id else None

        # If no video_id given, find any video for this room
        if not key or key not in self._state:
            for k in self._state:
                if k[0] == room_id:
                    key = k
                    break

        if not key or key not in self._state:
            return None

        state = self._state[key]

        if not state["queue"]:
            return None

        # Sort queue by start time (ascending) so we always get
        # the window closest to current_time first
        state["queue"].sort(key=lambda w: w["start"])

        window = state["queue"].pop(0)
        state["processing"] = True

        logger.info(
            "get_next_window — room=%s  %.1f → %.1fs  "
            "(queue remaining: %d, current_time: %.1fs)",
            room_id,
            window["start"],
            window["start"] + window["duration"],
            len(state["queue"]),
            state["current_time"],
        )

        return window

    # =========================================================================
    # PUBLIC — REFILL QUEUE  (🆕 Step 5: explicit refill trigger)
    # =========================================================================

    def refill_queue(
        self,
        room_id:  str,
        video_id: str = "",
    ) -> None:
        """
        Ensure the queue has enough windows to stay ahead of playback.

        Step 5: Called explicitly by the orchestrator after every
        window is processed, and when playback position updates.

        This is a thin wrapper around _refill_queue() that finds
        the correct state key.
        """

        # Try with video_id first
        key = (room_id, video_id) if video_id else None

        # If no video_id given, find any video for this room
        if not key or key not in self._state:
            for k in self._state:
                if k[0] == room_id:
                    key = k
                    break

        if not key or key not in self._state:
            logger.warning(
                "refill_queue: unknown room/video — room=%s video=%s",
                room_id, video_id,
            )
            return

        self._refill_queue(key)

        logger.debug(
            "refill_queue — room=%s  queue_length=%d",
            room_id, len(self._state[key]["queue"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

processing_queue = ProcessingQueue()