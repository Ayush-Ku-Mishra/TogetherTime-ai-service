"""
Subtitle Chunker

Combines Whisper subtitle segments into
scene-sized transcript chunks.

Input:
------
[
    {
        "text": "...",
        "timestamp": 120,
        "duration": 5
    },
    ...
]

Output:
-------
[
    {
        "start":120,
        "end":150,
        "duration":30,
        "text":"..."
    }
]
"""

from typing import Dict, List


class SubtitleChunker:

    def __init__(
        self,
        chunk_duration: float = 30.0,
    ):
        self.chunk_duration = chunk_duration

    def build_chunks(
        self,
        subtitles: List[Dict],
    ) -> List[Dict]:

        if not subtitles:
            return []

        chunks = []

        current = []

        chunk_start = subtitles[0]["timestamp"]

        for subtitle in subtitles:

            ts = subtitle["timestamp"]

            if ts - chunk_start >= self.chunk_duration:

                chunks.append(
                    self._create_chunk(current)
                )

                current = []

                chunk_start = ts

            current.append(subtitle)

        if current:

            chunks.append(
                self._create_chunk(current)
            )

        return chunks

    def _create_chunk(
        self,
        subtitles: List[Dict],
    ) -> Dict:

        start = subtitles[0]["timestamp"]

        last = subtitles[-1]

        end = last["timestamp"] + last["duration"]

        text = " ".join(

            s["text"].strip()

            for s in subtitles

            if s["text"].strip()

        )

        return {

            "start": round(start, 2),

            "end": round(end, 2),

            "duration": round(end - start, 2),

            "text": text,

        }


subtitle_chunker = SubtitleChunker()