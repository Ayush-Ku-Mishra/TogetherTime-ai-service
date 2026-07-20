# ai-service/services/llm_service.py

import os
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Priority order — fastest / most capable first
# When one fails with 429, automatically try next
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]


class LLMService:

    def __init__(self):
        self.models:            list  = []   # all working models
        self.gemini_model_name: str   = ""   # current primary model name
        self._quota_failed:     set   = set() # models that hit 429

        if not GEMINI_KEY:
            print("❌ GEMINI_API_KEY not set")
            return

        self._init_models()

    def _init_models(self):
        """
        Try all models in priority order.
        Keep all that work — not just the first.
        Primary = first working model.
        """

        for model_name in GEMINI_MODELS:
            try:
                m = genai.GenerativeModel(model_name)
                # Quick smoke test
                resp = m.generate_content(
                    "Hi",
                    generation_config={"max_output_tokens": 5},
                )
                if resp.text:
                    self.models.append((model_name, m))
                    print(f"✅ Gemini model ready: {model_name}")
            except Exception as e:
                print(f"⚠️ Model '{model_name}' not available: {e}")

        if self.models:
            self.gemini_model_name = self.models[0][0]
            print(f"🎯 Primary model: {self.gemini_model_name}")
            print(f"🔄 Fallback chain: {[m[0] for m in self.models]}")
        else:
            print("❌ No Gemini models available")

    # =========================================================================
    # PUBLIC — GENERATE
    # =========================================================================

    async def generate(self, prompt: str) -> tuple[str, str]:

        if not self.models:
            return (
                "❌ No AI model available. Please check your API key.",
                "none",
            )

        last_error = ""

        for model_name, model in self.models:

            if model_name in self._quota_failed:
                print(f"⏭️ Skipping {model_name} (quota exceeded)")
                continue

            try:
                print(f"🤖 Trying model: {model_name}")

                answer = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self._generate_sync,
                        model,
                        prompt,
                        1500,   # ← INCREASED from 500 to 1500
                    ),
                    timeout=30.0,   # ← INCREASED from 20s to 30s
                )

                if answer:
                    print(f"✅ {model_name} responded")
                    self._quota_failed.discard(model_name)
                    self.gemini_model_name = model_name
                    return answer, model_name

            except asyncio.TimeoutError:
                print(f"⏱️ {model_name} timed out — trying next")
                last_error = "timeout"
                continue

            except Exception as e:
                error_str = str(e).lower()

                if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                    print(
                        f"🚫 {model_name} quota exceeded (429) "
                        f"— marking failed, trying next model"
                    )
                    self._quota_failed.add(model_name)
                    last_error = "quota"
                    continue

                print(f"⚠️ {model_name} error: {e} — trying next")
                last_error = str(e)
                continue

        all_quota = len(self._quota_failed) == len(self.models)

        if all_quota:
            return (
                "⚠️ I've hit the API limit for now. "
                "Please wait a minute and try again! 🙏",
                "quota_exceeded",
            )

        return (
            "I'm having trouble connecting right now. "
            "Please try asking again in a moment.",
            "none",
        )

    def _generate_sync(
        self,
        model:      genai.GenerativeModel,
        prompt:     str,
        max_tokens: int = 1500,
    ) -> str:
        """
        Synchronous generation with full safety handling.
        
        Handles:
        - Blocked prompts (PROHIBITED_CONTENT)
        - Empty candidates
        - Safety filter false positives
        - Partial responses
        """

        # ── Safety settings ────────────────────────────────────────────────────
        # BLOCK_ONLY_HIGH reduces false positives for normal video content
        # dating videos, action movies, political discussions etc.
        safety_settings = [
            {
                "category":  "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category":  "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature":       0.7,
                },
                safety_settings=safety_settings,
            )

        except Exception as e:
            print(f"⚠️ generate_content failed: {e}")
            raise

        # ── Check if response was blocked ─────────────────────────────────────
        if not response.candidates:
            feedback         = getattr(response, "prompt_feedback", None)
            block_reason_str = "UNKNOWN"

            if feedback:
                block_reason     = getattr(feedback, "block_reason", None)
                block_reason_str = str(block_reason) if block_reason else "UNKNOWN"
                print(f"⚠️ Gemini blocked prompt — reason: {block_reason_str}")

            # ── If blocked because of VIDEO CONTEXT content ────────────────────
            # Try again with CLEAN prompt (no video context)
            # Just answer the question directly
            if "PROHIBITED" in block_reason_str or "SAFETY" in block_reason_str:
                print("🔄 Retrying with clean prompt (no video context)...")
                clean_response = self._generate_without_context(
                    model=model,
                    original_prompt=prompt,
                    max_tokens=max_tokens,
                )
                if clean_response:
                    return clean_response

            return (
                "I had trouble processing that with the current video context. "
                "Could you ask again? 😊"
            )

        # ── Extract text safely ────────────────────────────────────────────────
        try:
            text = response.text
            if text and text.strip():
                return text.strip()

        except Exception as e:
            print(f"⚠️ response.text access failed: {e}")

            # Try manual extraction from parts
            try:
                candidate = response.candidates[0]
                parts     = candidate.content.parts

                if parts:
                    extracted = "".join(
                        p.text for p in parts
                        if hasattr(p, "text") and p.text
                    ).strip()

                    if extracted:
                        return extracted

            except Exception as e2:
                print(f"⚠️ Manual extraction also failed: {e2}")

        return ""

    def _generate_without_context(
        self,
        model:           genai.GenerativeModel,
        original_prompt: str,
        max_tokens:      int = 1500,
    ) -> str:
        """
        Extract just the user question from the full prompt
        and answer it WITHOUT the video context that got blocked.

        This is the fallback when the full prompt is blocked.
        """

        # ── Extract user question from prompt ─────────────────────────────────
        # Our prompts always end with:
        # "USER QUESTION\n...\n{question}\n...YOUR ANSWER"
        question = ""

        try:
            # Try to find the question section
            if "USER QUESTION" in original_prompt:
                parts    = original_prompt.split("USER QUESTION")
                question = parts[-1].strip()

                # Remove the "YOUR ANSWER" part if present
                if "YOUR ANSWER" in question:
                    question = question.split("YOUR ANSWER")[0].strip()

                # Clean up separators
                import re
                question = re.sub(r"[═\n]+", " ", question).strip()

            elif "USER'S QUESTION" in original_prompt:
                parts    = original_prompt.split("USER'S QUESTION")
                question = parts[-1].strip()

        except Exception:
            pass

        if not question:
            return ""

        print(f"🔄 Clean prompt question: {question[:80]}")

        # ── Build minimal clean prompt ─────────────────────────────────────────
        clean_prompt = f"""
    You are TogetherTime AI — a friendly watch party companion.

    The user is watching a video and asked you this question:

    "{question}"

    Answer helpfully and naturally. If the question is about:
    - General knowledge → answer it fully
    - The video content → say you had trouble reading 
    the scene data but will try again soon
    - Personal/conversational → respond warmly

    Keep your answer concise and friendly.
    Use emojis naturally. 😊
    """

        safety_settings = [
            {
                "category":  "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category":  "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]

        try:
            response = model.generate_content(
                clean_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature":       0.7,
                },
                safety_settings=safety_settings,
            )

            if response.candidates:
                text = response.text
                if text and text.strip():
                    print("✅ Clean prompt succeeded!")
                    return text.strip()

        except Exception as e:
            print(f"⚠️ Clean prompt also failed: {e}")

        return ""


    # =========================================================================
    # PUBLIC — VISION
    # =========================================================================

    async def generate_with_vision(
        self,
        prompt:       str,
        image_base64: str,
    ) -> tuple[str, str]:

        if not self.models:
            return "Vision not available.", "none"

        try:
            import base64
            from PIL import Image
            import io

            image_bytes = base64.b64decode(image_base64)
            image       = Image.open(io.BytesIO(image_bytes))

            # Use primary model for vision
            _, model = self.models[0]

            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: model.generate_content(
                        [prompt, image],
                        generation_config={"max_output_tokens": 400},
                    ).text.strip()
                ),
                timeout=20.0,
            )
            return response, f"{self.models[0][0]}-vision"

        except Exception as e:
            print(f"⚠️ Vision failed: {e}")
            return await self.generate(prompt)

    # =========================================================================
    # PUBLIC — STATUS
    # =========================================================================

    @property
    def model(self):
        """Backwards compat — returns first model object or None."""
        return self.models[0][1] if self.models else None

    def status(self) -> dict:
        return {
            "primary":       self.gemini_model_name,
            "available":     [m[0] for m in self.models],
            "quota_failed":  list(self._quota_failed),
            "working":       [
                m[0] for m in self.models
                if m[0] not in self._quota_failed
            ],
        }


# Global singleton
llm_service = LLMService()