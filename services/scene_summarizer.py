# ai-service/services/scene_summarizer.py
"""
Scene Summarizer

Converts a transcript chunk into a scene object.

Scene Object
------------
{
    start:      float   — chunk start time in seconds
    end:        float   — chunk end time in seconds
    duration:   float   — chunk duration in seconds
    summary:    str     — LLM-generated English summary
    transcript: str     — original transcript text (preserved for embedding)
    keywords:   list    — key words extracted from transcript
    model:      str     — Gemini model used
}

Why we preserve transcript
--------------------------
The transcript is stored alongside the summary so that
Qdrant can embed BOTH when indexing.

If a scene says "in front of the elephants" in the transcript
but the summary says "the speaker describes an animal",
the query "elephant" will still match — because the transcript
is part of the embedding text.
"""

from typing import Dict, List, Optional

from .llm_service import llm_service


class SceneSummarizer:

    def __init__(self):
        self.enabled = llm_service.model is not None

        if self.enabled:
            print("✅ Scene Summarizer ready")
        else:
            print("⚠️ Scene Summarizer disabled (LLM unavailable)")

    # =========================================================================
    # PUBLIC
    # =========================================================================

    async def summarize(self, chunk: Dict) -> Optional[Dict]:
        """
        Convert a subtitle chunk into a scene object.

        Parameters
        ----------
        chunk : { start, end, duration, text }

        Returns
        -------
        {
            start, end, duration,
            summary, transcript, keywords,
            model
        }
        or None if summarization fails.
        """

        if not self.enabled:
            return None

        transcript = chunk.get("text", "").strip()

        if not transcript:
            return None

        prompt = self._build_prompt(transcript)

        try:
            summary, model = await llm_service.generate(prompt)
            summary        = summary.strip()

            if not summary:
                return None

            keywords = self._extract_keywords(transcript)

            return {
                "start":      chunk["start"],
                "end":        chunk["end"],
                "duration":   chunk["duration"],
                "summary":    summary,
                "transcript": transcript,       # ← preserved for Qdrant embedding
                "keywords":   keywords,
                "model":      model,
            }

        except Exception as e:
            print(f"⚠️ Scene summarization failed: {e}")
            return None

    # =========================================================================
    # PRIVATE
    # =========================================================================

    def _build_prompt(self, transcript: str) -> str:
        return f"""
You are helping an AI companion understand what is happening in a video.

Summarize ONLY the events in this transcript.

Rules
-----
- Maximum 3 sentences.
- Keep important actions and people.
- Do NOT invent names or facts.
- Do NOT add opinions or explanations.
- If the transcript is in Hindi, Odia, Telugu, Tamil or any other language,
  understand it and summarize in natural English.
- Write a factual, concise scene summary.

Transcript
----------
{transcript}
"""

    def _extract_keywords(self, transcript: str) -> List[str]:
        """
        Extract simple keywords from transcript text.

        These are added to the Qdrant embedding to improve
        retrieval quality for specific terms.

        Simple approach: lowercase words longer than 4 chars,
        deduplicated, stop words removed.
        """

        STOP_WORDS = {
            "that", "this", "with", "from", "they", "them",
            "their", "have", "been", "were", "will", "would",
            "could", "should", "about", "which", "when", "where",
            "what", "just", "then", "than", "there", "here",
            "some", "also", "into", "onto", "over", "under",
            "very", "really", "right", "okay", "actually",
        }

        words = transcript.lower().split()

        keywords = []
        seen     = set()

        for word in words:
            # Strip punctuation
            clean = "".join(c for c in word if c.isalpha())

            if (
                len(clean) > 4
                and clean not in STOP_WORDS
                and clean not in seen
            ):
                keywords.append(clean)
                seen.add(clean)

        return keywords[:20]  # cap at 20 keywords


scene_summarizer = SceneSummarizer()