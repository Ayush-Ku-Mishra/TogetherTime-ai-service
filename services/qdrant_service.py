# ai-service/services/qdrant_service.py
"""
Qdrant Service

Owns all vector database operations.

Embedding strategy
------------------
We embed Summary + Transcript + Keywords together so that
semantic search finds scenes based on ALL available text,
not just the summary alone.

Similarity threshold
--------------------
BGE-small-en-v1.5 with cosine similarity — observed ranges:

  Strong match   (topic present in video)  : 0.75 - 1.00
  Good match     (related content)         : 0.60 - 0.75
  Weak match     (tangentially related)    : 0.45 - 0.60
  Unrelated      (different topic)         : 0.30 - 0.50

Threshold = 0.55

  ✅ Passes  : "meditation awareness breathing"  → 0.80
  ✅ Passes  : "what happened in the video"      → 0.62
  ❌ Rejects : "rocket launch on the moon"       → 0.52
  ❌ Rejects : "elephant trunk zoo" (wrong video)→ 0.50

Override with QDRANT_THRESHOLD env var if needed.
"""

import os
import uuid
from typing import Dict, List, Optional

from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# ── Threshold ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("QDRANT_THRESHOLD", "0.55"))


class QdrantService:

    def __init__(self):

        self.client          = None
        self.encoder         = None
        self.enabled         = False
        self.collection_name = "watch_party_memory"
        self.vector_size     = 384

        # ── Embedding model ────────────────────────────────────────────────
        try:
        except Exception as e:
            print(f"⚠️ Failed to load embedding model: {e}")
            return

        # ── Qdrant client ──────────────────────────────────────────────────
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            qdrant_url = os.getenv("QDRANT_URL", "")
            qdrant_key = os.getenv("QDRANT_API_KEY", "")

            if not qdrant_url:
                print("⚠️ QDRANT_URL not configured")
                return

            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_key or None,
                timeout=30,
            )

            self.Distance     = Distance
            self.VectorParams = VectorParams
            self.enabled      = True

            print(f"✅ Qdrant connected  (threshold={SIMILARITY_THRESHOLD})")

        except Exception as e:
            print(f"⚠️ Qdrant connection failed: {e}")

    # =========================================================================
    # COLLECTION
    # =========================================================================

    async def init_collections(self):

        if not self.enabled:
            return

        try:
            from qdrant_client.http.models import PayloadSchemaType

            existing = [
                c.name
                for c in self.client.get_collections().collections
            ]

            if self.collection_name in existing:
                print("🗑 Removing old collection...")
                self.client.delete_collection(self.collection_name)

            print("📦 Creating collection...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.vector_size,
                    distance=self.Distance.COSINE,
                ),
            )
            print("✅ Collection created")

            indexes = [
                ("roomId",   PayloadSchemaType.KEYWORD),
                ("videoId",  PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
                ("source",   PayloadSchemaType.KEYWORD),
                ("start",    PayloadSchemaType.FLOAT),
                ("end",      PayloadSchemaType.FLOAT),
            ]

            for field, schema in indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                    print(f"  ✅ Index: {field}")
                except Exception:
                    pass

            print("✅ Qdrant ready")

        except Exception as e:
            print(f"⚠️ Collection init failed: {e}")

    def ensure_encoder_loaded(self):
        """
        Load embedding model only when first needed.
        """
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            print("✅ SentenceTransformer loaded")

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    def _build_embedding_text(self, scene: Dict) -> str:
        """
        Combine Summary + Transcript + Keywords.

        This ensures all content words end up in the vector —
        not just what the LLM chose to put in the summary.
        """

        summary      = (scene.get("summary",    "") or "").strip()
        transcript   = (scene.get("transcript", "") or "").strip()
        keywords     = scene.get("keywords", []) or []
        keyword_text = " ".join(str(k) for k in keywords)

        parts = []

        if summary:
            parts.append(f"Summary\n{summary}")

        if transcript:
            parts.append(f"Transcript\n{transcript}")

        if keyword_text:
            parts.append(f"Keywords\n{keyword_text}")

        return "\n\n".join(parts) if parts else summary

    def _embed(self, text: str) -> List[float]:
        self.ensure_encoder_loaded()

        vector = self.encoder.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    # =========================================================================
    # STORE
    # =========================================================================

    async def store_scene(self, room_id: str, scene: Dict):

        if not self.enabled:
            return

        try:
            from qdrant_client.http.models import PointStruct

            embedding_text = self._build_embedding_text(scene)

            if not embedding_text.strip():
                print("⚠️ Empty embedding text — skipping")
                return

            vector = self._embed(embedding_text)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "roomId":     room_id,
                    "videoId":    scene.get("video_id",    ""),
                    "start":      scene.get("start"),
                    "end":        scene.get("end"),
                    "duration":   scene.get("duration"),
                    "language":   scene.get("language",    "unknown"),
                    "summary":    scene.get("summary",     ""),
                    "transcript": scene.get("transcript",  ""),
                    "keywords":   scene.get("keywords",    []),
                    "model":      scene.get("model",       ""),
                    "source":     scene.get("source",      "whisper"),
                },
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            print(
                f"✅ Stored scene "
                f"{scene.get('start')} → {scene.get('end')} | "
                f"embed_len={len(embedding_text)}"
            )

        except Exception as e:
            print(f"⚠️ Store scene failed: {e}")

    # =========================================================================
    # SEMANTIC SEARCH  (with threshold)
    # =========================================================================

    async def search(
        self,
        room_id: str,
        video_id: str,
        query:   str,
        limit:   int = 5,
    ) -> List[Dict]:
        """
        Semantic search.  Results below SIMILARITY_THRESHOLD are discarded.

        Returns [] when no relevant scene exists — do NOT force a result.
        """

        if not self.enabled:
            return []

        try:
            from qdrant_client.http.models import (
                Filter, FieldCondition, MatchValue,
            )

            vector = self._embed(query)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="roomId",
                            match=MatchValue(value=room_id),
                        ),
                        FieldCondition(
                            key="videoId",
                            match=MatchValue(value=video_id),
                        ),
                    ]
                ),
                score_threshold=SIMILARITY_THRESHOLD,
            )

            scenes = []
            for hit in results:
                payload          = dict(hit.payload)
                payload["score"] = round(hit.score, 4)
                scenes.append(payload)

            if not scenes:
                print(
                    f"⚠️ No scene above threshold={SIMILARITY_THRESHOLD} "
                    f"for query: '{query[:50]}'"
                )

            return scenes

        except Exception as e:
            print(f"⚠️ Semantic search failed: {e}")
            return []

    # =========================================================================
    # RAW SEARCH  (no threshold — for debugging only)
    # =========================================================================

    async def search_raw(
        self,
        room_id: str,
        query:   str,
        limit:   int = 5,
    ) -> List[Dict]:
        """
        Search with NO threshold.
        Use only in debug scripts and tests — never in production queries.
        """

        if not self.enabled:
            return []

        try:
            from qdrant_client.http.models import (
                Filter, FieldCondition, MatchValue,
            )

            vector = self._embed(query)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=limit,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="roomId",
                            match=MatchValue(value=room_id),
                        )
                    ]
                ),
            )

            scenes = []
            for hit in results:
                payload          = dict(hit.payload)
                payload["score"] = round(hit.score, 4)
                scenes.append(payload)

            return scenes

        except Exception as e:
            print(f"⚠️ search_raw failed: {e}")
            return []

    async def ensure_collection(self):
        """
        Create collection ONLY if it does not exist.
        
        NEVER deletes existing data.
        Called on startup instead of init_collections().
        
        init_collections() still exists for manual reset only.
        """

        if not self.enabled:
            return

        try:
            existing = [
                c.name
                for c in self.client.get_collections().collections
            ]

            if self.collection_name in existing:
                # ── Collection exists — just verify it ────────────────────────
                info = self.client.get_collection(self.collection_name)
                count = info.points_count
                print(
                    f"✅ Qdrant collection exists — "
                    f"{count} vectors preserved ✅"
                )
                return

            # ── Collection does not exist — create fresh ──────────────────────
            print("📦 Creating Qdrant collection (first time)...")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.vector_size,
                    distance=self.Distance.COSINE,
                ),
            )

            # Create indexes
            from qdrant_client.http.models import PayloadSchemaType

            indexes = [
                ("roomId",   PayloadSchemaType.KEYWORD),
                ("videoId",  PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
                ("source",   PayloadSchemaType.KEYWORD),
                ("start",    PayloadSchemaType.FLOAT),
                ("end",      PayloadSchemaType.FLOAT),
            ]

            for field, schema in indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                    print(f"  ✅ Index: {field}")
                except Exception:
                    pass

            print("✅ Qdrant collection created fresh")

        except Exception as e:
            print(f"⚠️ ensure_collection failed: {e}")

    # =========================================================================
    # TIMESTAMP SEARCH  (no threshold)
    # =========================================================================

    async def search_near_timestamp(
        self,
        room_id:   str,
        video_id:  str,
        timestamp: float,
    ) -> Optional[Dict]:
        """
        Return the scene nearest to timestamp.

        Strategy
        --------
        1. Try exact match  (timestamp falls inside scene start→end)
        2. Try nearest match (closest scene by start time)
        — handles gaps between scenes
        """

        if not self.enabled:
            return None

        try:

            all_scenes = await self.get_video_scenes(
                room_id=room_id,
                video_id=video_id,
            )

            if not all_scenes:
                return None

            # ── Pass 1: exact containment ──────────────────────────────────
            for scene in all_scenes:
                if scene.get("start", 0) <= timestamp <= scene.get("end", 0):
                    return scene

            # ── Pass 2: nearest scene by distance ─────────────────────────
            # Handles gaps — find the scene whose midpoint is closest
            def distance(scene):
                mid = (scene.get("start", 0) + scene.get("end", 0)) / 2
                return abs(mid - timestamp)

            nearest = min(all_scenes, key=distance)

            logger.info(
                "search_near_timestamp: no exact match at %.1fs, "
                "using nearest scene %.1f→%.1f",
                timestamp,
                nearest.get("start", 0),
                nearest.get("end", 0),
            )

            return nearest

        except Exception:
            logger.exception("search_near_timestamp failed")
            return None

    # =========================================================================
    # GET VIDEO SCENES
    # =========================================================================

    async def get_video_scenes(
        self,
        room_id:  str,
        video_id: str,
    ) -> List[Dict]:

        if not self.enabled:
            return []

        try:
            from qdrant_client.http.models import (
                Filter, FieldCondition, MatchValue,
            )

            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="roomId",
                            match=MatchValue(value=room_id),
                        ),
                        FieldCondition(
                            key="videoId",
                            match=MatchValue(value=video_id),
                        ),
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )

            scenes = [point.payload for point in results]
            scenes.sort(key=lambda x: x.get("start", 0))

            return scenes

        except Exception as e:
            print(f"⚠️ Get video scenes failed: {e}")
            return []

    # =========================================================================
    # DELETE
    # =========================================================================

    async def delete_video(self, room_id: str, video_id: str):

        if not self.enabled:
            return

        try:
            from qdrant_client.http.models import (
                Filter, FieldCondition, MatchValue,
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="roomId",  match=MatchValue(value=room_id)),
                        FieldCondition(key="videoId", match=MatchValue(value=video_id)),
                    ]
                ),
            )
            print(f"🗑 Deleted memory for video: {video_id}")

        except Exception as e:
            print(f"⚠️ Delete video failed: {e}")

    # =========================================================================
    # PUBLIC — GET CURRENT SCENE  (for VideoContextService)
    # =========================================================================

    async def get_current_scene(
        self,
        room_id:   str,
        video_id:  str,
        timestamp: float,
    ) -> Optional[Dict]:
        """
        Return the scene whose time range contains the given timestamp.

        Wrapper around search_near_timestamp() for VideoContextService.

        Parameters
        ----------
        room_id   : watch party room identifier
        video_id  : YouTube video id
        timestamp : current playback position in seconds

        Returns
        -------
        Scene dict, or None if no scene found near timestamp.
        """

        return await self.search_near_timestamp(
            room_id=room_id,
            video_id=video_id,
            timestamp=timestamp,
        )


    # =========================================================================
    # PUBLIC — GET RECENT SCENES  (for VideoContextService)
    # =========================================================================

    async def get_recent_scenes(
        self,
        room_id:  str,
        video_id: str,
        before:   float,
        limit:    int = 5,
    ) -> List[Dict]:
        """
        Return the last N scenes before a given timestamp.

        Used to provide recent context to the AI without a semantic query.

        Parameters
        ----------
        room_id  : watch party room identifier
        video_id : YouTube video id
        before   : timestamp in seconds (only return scenes that end before this)
        limit    : maximum number of scenes to return

        Returns
        -------
        List of scene dicts, ordered chronologically (earliest first).
        Empty list if no scenes found.

        Example
        -------
        before = 120s
        limit  = 5

        Returns scenes ending before 120s:
        [60-90, 90-120]  (not [120-150] because it ends after 120s)
        """

        if not self.enabled:
            return []

        try:
            # Get all scenes for this video
            all_scenes = await self.get_video_scenes(
                room_id=room_id,
                video_id=video_id,
            )

            if not all_scenes:
                return []

            # Filter: only scenes that END before the timestamp
            filtered = [
                s for s in all_scenes
                if s.get("end", 0) <= before
            ]

            # Sort chronologically (earliest first)
            filtered.sort(key=lambda s: s.get("start", 0))

            # Take last N
            recent = filtered[-limit:] if len(filtered) > limit else filtered

            logger.info(
                "get_recent_scenes — room=%s  before=%.1fs  found=%d",
                room_id, before, len(recent),
            )

            return recent

        except Exception:
            logger.exception("get_recent_scenes failed")
            return []

    async def delete_room(self, room_id: str):

        if not self.enabled:
            return

        try:
            from qdrant_client.http.models import (
                Filter, FieldCondition, MatchValue,
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="roomId", match=MatchValue(value=room_id)),
                    ]
                ),
            )
            print(f"🗑 Deleted room memory: {room_id}")

        except Exception as e:
            print(f"⚠️ Delete room failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

qdrant_service = QdrantService()