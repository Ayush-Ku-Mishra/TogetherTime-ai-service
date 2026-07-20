# ai-service/services/subtitle_pipeline.py
"""
Auto subtitle extraction and scene summary pipeline.
This runs continuously during a watch session.
Every 30 seconds of subtitles → generate scene summary → store in Qdrant
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime


class SubtitlePipeline:
    """
    Manages the rolling subtitle buffer and triggers scene summary generation.
    Called by the Node.js backend every time new subtitles are available.
    """

    def __init__(self):
        self.subtitle_buffers: Dict[str, List[dict]] = {}  # roomId -> subtitles
        self.last_summary_time: Dict[str, float] = {}      # roomId -> timestamp
        self.SUMMARY_INTERVAL_SECONDS = 30  # Generate summary every 30s of content
        self.MIN_SUBTITLES_FOR_SUMMARY = 3  # Need at least 3 subtitle lines

    def add_subtitles(self, room_id: str, subtitles: List[dict]) -> List[dict]:
        """
        Add new subtitles to the room buffer.
        Returns subtitles ready for summary if threshold reached.
        """
        if room_id not in self.subtitle_buffers:
            self.subtitle_buffers[room_id] = []
            self.last_summary_time[room_id] = 0

        # Add new subtitles (avoid duplicates by timestamp)
        existing_timestamps = {s.get("timestamp", 0) for s in self.subtitle_buffers[room_id]}
        for sub in subtitles:
            ts = sub.get("timestamp", 0)
            if ts not in existing_timestamps:
                self.subtitle_buffers[room_id].append(sub)
                existing_timestamps.add(ts)

        # Sort by timestamp
        self.subtitle_buffers[room_id].sort(key=lambda x: x.get("timestamp", 0))

        # Keep buffer manageable (last 100 subtitles)
        if len(self.subtitle_buffers[room_id]) > 100:
            self.subtitle_buffers[room_id] = self.subtitle_buffers[room_id][-100:]

        # Check if we should generate a summary
        return self._get_subtitles_for_summary(room_id)

    def _get_subtitles_for_summary(self, room_id: str) -> List[dict]:
        """
        Returns subtitle batch if enough time has passed since last summary.
        """
        buffer = self.subtitle_buffers.get(room_id, [])
        if len(buffer) < self.MIN_SUBTITLES_FOR_SUMMARY:
            return []

        # Check time gap
        if not buffer:
            return []

        latest_ts = buffer[-1].get("timestamp", 0)
        last_summary_ts = self.last_summary_time.get(room_id, 0)

        if latest_ts - last_summary_ts >= self.SUMMARY_INTERVAL_SECONDS:
            # Get subtitles since last summary
            batch = [s for s in buffer if s.get("timestamp", 0) > last_summary_ts]
            if len(batch) >= self.MIN_SUBTITLES_FOR_SUMMARY:
                self.last_summary_time[room_id] = latest_ts
                return batch

        return []

    def get_recent_subtitles(self, room_id: str, last_n: int = 10) -> List[dict]:
        """Get the most recent N subtitles for a room."""
        buffer = self.subtitle_buffers.get(room_id, [])
        return buffer[-last_n:] if buffer else []

    def clear_room(self, room_id: str):
        """Clear all subtitle data for a room."""
        self.subtitle_buffers.pop(room_id, None)
        self.last_summary_time.pop(room_id, None)


# Global instance
subtitle_pipeline = SubtitlePipeline()