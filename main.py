# ai-service/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

from routers.ai_router import router as ai_router

# ✅ Use lifespan instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    grok_key   = os.getenv("GROK_API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "")
    redis_url  = os.getenv("REDIS_URL", "")

    print("=" * 50)
    print("🚀 TogetherTime AI Service Starting...")
    print(f"   Gemini Key : {'✅ SET' if gemini_key else '❌ MISSING'}")
    print(f"   Grok Key   : {'✅ SET' if grok_key   else '❌ MISSING'}")
    print(f"   Qdrant URL : {'✅ SET' if qdrant_url else '❌ MISSING'}")
    print(f"   Redis URL  : {'✅ SET' if redis_url  else '❌ MISSING'}")
    print(f"   Port       : {os.getenv('AI_SERVICE_PORT', 8001)}")
    print("=" * 50)

    # ── Qdrant: ONLY create if not exists — NEVER delete on restart ────────
    try:
        from services.qdrant_service import QdrantService
        qdrant = QdrantService()
        await qdrant.ensure_collection()   # ← NEW method (see Fix 2)
    except Exception as e:
        print(f"⚠️ Qdrant startup warning (non-fatal): {e}")

    # ── Start 24h cleanup background job ──────────────────────────────────
    from services.cleanup_service import cleanup_service
    cleanup_task = asyncio.create_task(
        cleanup_service.run_cleanup_loop()
    )
    print("✅ 24h auto-cleanup job started")

    yield

    print("🛑 TogetherTime AI Service shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    print("✅ Cleanup job stopped")


app = FastAPI(
    title="TogetherTime AI Service",
    description="AI Companion for Watch Parties",
    version="1.0.0",
    lifespan=lifespan   # ✅ Use lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router, prefix="/ai", tags=["AI"])

@app.get("/health")
async def health():
    return {
        "status":  "healthy",
        "service": "TogetherTime AI",
        "gemini":  bool(os.getenv("GEMINI_API_KEY")),
        "grok":    bool(os.getenv("GROK_API_KEY")),
    }

if __name__ == "__main__":
    port = int(os.getenv("AI_SERVICE_PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )