# ai-service/routers/ai_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from models.ai_models import (
    AIRequest, AIResponse,
    RecommendRequest, SessionSummaryRequest, TranslateRequest, ProcessVideoRequest,
)
from services.llm_service           import llm_service
from services.context_builder       import context_builder
from services.youtube_service       import get_video_title
from services.scene_memory          import scene_memory
from services.subtitle_pipeline     import subtitle_pipeline
from services.pipeline_orchestrator import pipeline_orchestrator
from services.processing_queue      import processing_queue
from services.video_context         import video_context
from services.conversation_memory   import conversation_memory
from services.ai_orchestrator       import ai_orchestrator
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

_title_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────

class SubtitleBatchRequest(BaseModel):
    roomId:     str
    movieTitle: Optional[str] = "Unknown"
    videoId:    Optional[str] = ""
    platform:   Optional[str] = "youtube"
    timestamp:  float         = 0
    subtitles:  List[dict]    = []


class SubtitleQueryRequest(BaseModel):
    roomId: str
    lastN:  int = 10


class PlaybackUpdateRequest(BaseModel):
    roomId:      str
    videoId:     str   = ""
    currentTime: float


# ─────────────────────────────────────────────────────────────────────────────
# Helper — resolve movie title
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_movie_title(request: AIRequest) -> str:

    movie_title = request.movieTitle or ""
    video_id    = None
    platform    = "youtube"

    if "video (ID:" in movie_title:
        try:
            parts    = movie_title.split("(ID:")
            platform = parts[0].strip().split()[0].lower()
            video_id = parts[1].replace(")", "").strip()
        except Exception:
            pass

    cache_key = f"{platform}:{video_id}"

    if cache_key in _title_cache:
        return _title_cache[cache_key]

    if video_id:
        title = await get_video_title(platform, video_id)
        _title_cache[cache_key] = title
        return title

    return movie_title or "the content being watched"


# ─────────────────────────────────────────────────────────────────────────────
# /ask
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AIResponse)
async def ask_ai(request: AIRequest):
    try:
        # ── Resolve real title ────────────────────────────────────────────
        real_title         = await resolve_movie_title(request)
        request.movieTitle = real_title

        # ── Extract identifiers ───────────────────────────────────────────
        video_id = request.movieId or ""
        platform = "youtube"

        # ── AI Orchestrator handles everything ────────────────────────────
        result = await ai_orchestrator.answer(
            room_id=request.roomId,
            video_id=video_id,
            platform=platform,
            question=request.userQuestion,
            user_id=getattr(request, "userId", ""),
            current_timestamp=request.currentTimestamp,
        )

        return AIResponse(
            answer=result["answer"],
            model_used=result["model"],
            success=result["success"],
        )

    except Exception as e:
        logger.exception("/ask failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /subtitles/push
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/subtitles/push")
async def push_subtitles(request: SubtitleBatchRequest):

    try:
        result = await pipeline_orchestrator.process_subtitle_push(
            room_id=request.roomId,
            video_id=request.videoId or "",
            timestamp=request.timestamp or 0,
            platform=request.platform or "youtube",
            movie_title=request.movieTitle or "",
            frontend_subs=request.subtitles or [],
        )

        return {
            "processed":       result["processed"],
            "captions_used":   result["subtitles_used"],
            "subtitle_source": result["subtitle_source"],
            "content_type":    result["content_type"],
            "summary_created": result["summary_created"],
            "summary":         result["summary"],
            "has_tmdb":        result["tmdb_data"] is not None,
        }

    except Exception as e:
        import traceback
        print(f"❌ /subtitles/push error: {e}")
        print(traceback.format_exc())
        return {"processed": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# /subtitles/{room_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/subtitles/{room_id}")
async def get_subtitles(room_id: str, last_n: int = 10):
    subtitles = subtitle_pipeline.get_recent_subtitles(room_id, last_n)
    return {"subtitles": subtitles, "count": len(subtitles)}


# ─────────────────────────────────────────────────────────────────────────────
# /memory/{room_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/memory/{room_id}")
async def clear_room_memory(room_id: str):
    from services.cleanup_service import cleanup_service
    result = await cleanup_service.cleanup_room(room_id)
    return {
        "cleared": True,
        "roomId":  room_id,
        "details": result,
    }



# ─────────────────────────────────────────────────────────────────────────────
# /recommend
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/recommend")
async def recommend_movies(request: RecommendRequest):

    try:
        current = (
            f"They are currently watching: {request.currentMovie}."
            if request.currentMovie else ""
        )

        prompt = f"""
You are a movie recommendation expert for a watch party group.
Group size: {request.groupSize} people.
{current}
Their preferences: {request.preferences}

Recommend exactly 5 movies perfect for this group.
For each movie include:
- Title and year
- Genre
- Why it matches their mood (1 line)
- One exciting description line

Use emojis and be enthusiastic!
"""
        answer, model = await llm_service.generate(prompt)

        return {
            "recommendations": answer,
            "model_used":      model,
            "success":         True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /translate
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/translate")
async def translate_dialogue(request: TranslateRequest):

    try:
        prompt = f"""
Translate this movie dialogue to {request.targetLanguage}:

"{request.text}"

{f'Scene context: {request.context}' if request.context else ''}

Provide:
1. 🔤 Direct translation
2. 💡 Meaning in context
3. 🌍 Cultural notes (if any)
"""
        answer, model = await llm_service.generate(prompt)

        return {
            "translated": answer,
            "model_used": model,
            "success":    True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /session-summary
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/session-summary")
async def session_summary(request: SessionSummaryRequest):

    try:
        scene_history = []

        try:
            from memory.rolling_memory import RollingMemory
            redis         = RollingMemory()
            scene_history = redis.get_history(request.roomId)
        except Exception:
            pass

        scene_text = (
            "\n".join([
                f"[{int(s.get('timestamp', 0) // 60)}m] {s.get('summary', '')}"
                for s in scene_history[-20:]
            ])
            if scene_history else "No scene data recorded"
        )

        chat = (
            "\n".join([
                f"- {m.get('user', 'User')}: {m.get('text', '')}"
                for m in request.chatHighlights[:15]
            ])
            if request.chatHighlights else "No chat recorded"
        )

        mins = request.watchDuration // 60
        secs = request.watchDuration % 60

        prompt = f"""
Create an exciting end-of-watch-party summary!

🎬 Content: {request.movieTitle}
⏱️ Duration: {mins}m {secs}s
👥 Participants: {request.participantCount}

📽️ Scene Timeline:
{scene_text}

💬 Chat Highlights:
{chat}

Include these sections:
🎬 Recap (3-4 sentences)
⭐ Memorable Moments
😂 Funniest Chat Moments
🎯 Fun Trivia (3 facts)
🎥 Watch Next (3 recommendations)
📊 Stats
🏆 Session Rating (/10)
"""
        answer, model = await llm_service.generate(prompt)

        # Clear room data
        pipeline_orchestrator.clear_room(request.roomId)

        try:
            from memory.rolling_memory import RollingMemory
            redis = RollingMemory()
            redis.clear(request.roomId)
        except Exception:
            pass

        return {
            "summary":    answer,
            "model_used": model,
            "success":    True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /process-video
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/process-video")
async def process_video(request: ProcessVideoRequest):

    result = await pipeline_orchestrator.process_video(
        room_id=request.roomId,
        video_id=request.videoId,
        movie_title=request.movieTitle,
        language=request.language,
    )

    return {
        "success": True,
        "result":  result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# /playback-update
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/playback-update")
async def playback_update(request: PlaybackUpdateRequest):
    try:
        # Simple — just update playback position
        processing_queue.update_playback(
            room_id=request.roomId,
            video_id=request.videoId or "",
            current_time=request.currentTime,
        )

        processed_until = processing_queue.get_processed_until(
            room_id=request.roomId,
            video_id=request.videoId or "",
        )

        buffer = processed_until - request.currentTime

        return {
            "success":        True,
            "currentTime":    request.currentTime,
            "processedUntil": processed_until,
            "bufferAhead":    round(buffer, 1),
        }

    except Exception as e:
        logger.exception("playback_update failed")
        return {
            "success": False,
            "error":   str(e),
        }

async def _resume_processing(
    room_id:  str,
    video_id: str,
) -> None:
    """
    Resume video processing after server restart.

    Called when:
    - State was restored from Redis
    - Video is not fully processed
    - No processing loop is running

    Uses cached audio (no re-download needed).
    Starts window loop from restored processed_until position.
    """

    logger.info(
        "🔄 _resume_processing — room=%s video=%s",
        room_id, video_id,
    )

    try:
        # Get saved metadata for platform info
        from services.state_persistence import state_persistence
        from services.video_metadata    import video_metadata

        # Try to get platform from saved metadata
        meta = state_persistence.load_metadata("youtube", video_id)
        platform = "youtube"  # default

        if not meta:
            # Try fetching fresh
            meta = await video_metadata.get_metadata(
                platform=platform,
                video_id=video_id,
            )

        movie_title = meta.get("title", "") if meta else ""

        logger.info(
            "_resume_processing: starting pipeline — "
            "room=%s video=%s title=%s",
            room_id, video_id, movie_title,
        )

        # Call process_video — it will:
        # 1. Find cached audio (no re-download)
        # 2. Call start_video (restores from Redis)
        # 3. Start window loop from restored position
        result = await pipeline_orchestrator.process_video(
            room_id=room_id,
            video_id=video_id,
            movie_title=movie_title,
            platform=platform,
        )

        logger.info(
            "_resume_processing: done — "
            "room=%s video=%s result=%s",
            room_id, video_id,
            {
                "success":  result.get("success"),
                "windows":  result.get("total_windows"),
                "scenes":   result.get("total_scenes"),
                "cached":   result.get("cached"),
            },
        )

    except Exception:
        logger.exception(
            "_resume_processing failed — room=%s video=%s",
            room_id, video_id,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Debug endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/debug/redis/{room_id}")
async def debug_redis(room_id: str):

    from memory.rolling_memory import RollingMemory
    redis   = RollingMemory()
    history = redis.get_history(room_id)

    return {
        "room_id":     room_id,
        "scene_count": len(history),
        "scenes":      history,
    }


@router.get("/debug/qdrant/{room_id}/{video_id}")
async def debug_qdrant(room_id: str, video_id: str):

    from services.qdrant_service import QdrantService

    qdrant = QdrantService()

    results = await qdrant.search(
        room_id=room_id,
        video_id=video_id,
        query="what happened in the scene",
        limit=5,
    )

    return {
        "room_id":      room_id,
        "video_id":     video_id,
        "result_count": len(results),
        "results":      results,
    }


@router.get("/debug/captions/{room_id}")
async def debug_captions(
    room_id:   str,
    video_id:  str   = "",
    timestamp: float = 60.0,
):
    from services.pipeline_orchestrator import get_captions, _caption_store

    stored = get_captions(
        room_id=room_id,
        video_id=video_id,
        current_timestamp=timestamp,
        window_seconds=60.0,
    )

    all_rooms = {
        rid: {vid: len(caps) for vid, caps in vids.items()}
        for rid, vids in _caption_store.items()
    }

    return {
        "room_id":        room_id,
        "video_id":       video_id,
        "timestamp":      timestamp,
        "found_captions": len(stored),
        "sample":         stored[:3] if stored else [],
        "all_rooms":      all_rooms,
    }


@router.get("/debug/live/{room_id}")
async def debug_live(room_id: str, video_id: str = ""):
    """Live monitoring dashboard for a room."""

    import time
    from memory.rolling_memory          import RollingMemory
    from services.pipeline_orchestrator import _caption_store
    from services.whisper_service       import whisper_service

    redis         = RollingMemory()
    scene_history = redis.get_history(room_id)

    recent_subs  = subtitle_pipeline.get_recent_subtitles(room_id, 10)
    buffer_size  = len(subtitle_pipeline.subtitle_buffers.get(room_id, []))
    last_summary = subtitle_pipeline.last_summary_time.get(room_id, 0)

    cap_store  = _caption_store.get(room_id, {})
    total_caps = sum(len(v) for v in cap_store.values())

    qdrant_results = []
    try:
        from services.qdrant_service import QdrantService
        qdrant         = QdrantService()
        qdrant_results = await qdrant.search(
            room_id=room_id,
            query="what happened",
            limit=5,
        )
    except Exception as e:
        qdrant_results = [{"error": str(e)}]

    return {
        "room_id":   room_id,
        "video_id":  video_id,
        "timestamp": time.time(),

        "redis": {
            "connected":   redis.enabled,
            "scene_count": len(scene_history),
            "scenes": [
                {
                    "time":    f"{int(s.get('timestamp',0)//60)}:{int(s.get('timestamp',0)%60):02d}",
                    "summary": s.get("summary", "")[:100],
                }
                for s in scene_history[-5:]
            ],
        },

        "qdrant": {
            "vector_count":   len(qdrant_results),
            "recent_vectors": [
                {
                    "time":    f"{int(r.get('timestamp',0)//60)}:{int(r.get('timestamp',0)%60):02d}",
                    "summary": r.get("summary", "")[:80],
                }
                for r in qdrant_results
            ],
        },

        "subtitle_pipeline": {
            "buffer_size":      buffer_size,
            "last_summary_at":  f"{int(last_summary//60)}:{int(last_summary%60):02d}",
            "recent_subtitles": recent_subs[-3:],
        },

        "caption_store": {
            "total_captions": total_caps,
            "videos_tracked": list(cap_store.keys()),
        },

        "whisper": {
            "enabled": whisper_service.enabled,
            "model":   whisper_service.model is not None,
        },

        "pipeline_health": {
            "redis_ok":        redis.enabled,
            "whisper_ok":      whisper_service.enabled,
            "scenes_stored":   len(scene_history) > 0,
            "captions_stored": total_caps > 0,
        },
    }


@router.get("/debug/pipeline/{room_id}")
async def debug_pipeline(room_id: str):
    """Full pipeline status for a room."""

    from memory.rolling_memory          import RollingMemory
    from services.pipeline_orchestrator import _caption_store

    redis         = RollingMemory()
    scene_history = redis.get_history(room_id)
    recent_subs   = subtitle_pipeline.get_recent_subtitles(room_id, 5)
    cap_store     = _caption_store.get(room_id, {})

    return {
        "room_id":       room_id,
        "redis_scenes":  len(scene_history),
        "recent_subs":   len(recent_subs),
        "caption_store": {
            vid: len(caps) for vid, caps in cap_store.items()
        },
        "last_scene":    scene_history[-1] if scene_history else None,
        "whisper_ready": False,
    }


@router.get("/debug/processing/{room_id}/{video_id}")
async def debug_processing(room_id: str, video_id: str):
    """Debug endpoint for processing queue."""

    try:
        status = processing_queue.status(room_id, video_id)

        return {
            "success":    True,
            "processing": pipeline_orchestrator.is_processing(room_id),
            "status":     status,
        }

    except Exception as e:
        logger.exception("debug_processing failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health")
async def ai_health():

    from services.whisper_service import whisper_service

    return {
        "status":    "operational",
        "model":     llm_service.gemini_model_name or "none",
        "whisper":   whisper_service.enabled,
        "endpoints": [
            "/ask",
            "/recommend",
            "/translate",
            "/session-summary",
            "/subtitles/push",
        ],
    }