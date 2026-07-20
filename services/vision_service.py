from .llm_service import LLMService

class VisionService:
    def __init__(self):
        self.llm = LLMService()

    async def analyze_frame(self, base64_image: str, question: str = None):
        prompt = question or "Describe this movie scene in detail. Identify any visible characters, objects, and emotions without giving spoilers."
        # In production you would decode base64 and pass as image part to Gemini
        response = await self.llm.generate(prompt, use_vision=True, image_data=base64_image)
        return {"description": response}