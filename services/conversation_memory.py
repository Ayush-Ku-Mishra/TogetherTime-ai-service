"""
Conversation Memory Service — with Redis persistence
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import redis as redis_lib
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_HISTORY: int = 20
TTL_SECONDS: int = 86400  # 24 hours


class ConversationMemory:
    """
    Conversation history with Redis persistence.
    
    Survives server restarts.
    Falls back to in-memory if Redis unavailable.
    """

    def __init__(self):
        self._conversations: Dict[str, List[Dict]] = {}
        self._redis        = None
        self._redis_enabled = False
        self._connect_redis()
        logger.info("✅ ConversationMemory initialized")

    # =========================================================================
    # PRIVATE — REDIS CONNECTION
    # =========================================================================

    def _connect_redis(self):
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis = redis_lib.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                ssl_cert_reqs=None,
            )
            self._redis.ping()
            self._redis_enabled = True
            logger.info("✅ ConversationMemory Redis connected")
        except Exception as e:
            logger.warning(
                "ConversationMemory: Redis unavailable — "
                "using in-memory only: %s", e
            )
            self._redis_enabled = False

    # =========================================================================
    # PRIVATE — REDIS HELPERS
    # =========================================================================

    def _redis_key(self, room_id: str) -> str:
        return f"conversation:{room_id}"

    def _save_to_redis(self, room_id: str) -> None:
        """Save current in-memory conversation to Redis."""
        if not self._redis_enabled:
            return
        try:
            key      = self._redis_key(room_id)
            messages = self._conversations.get(room_id, [])
            self._redis.set(
                key,
                json.dumps(messages),
                ex=TTL_SECONDS,
            )
        except Exception as e:
            logger.warning("ConversationMemory Redis save failed: %s", e)

    def _load_from_redis(self, room_id: str) -> List[Dict]:
        """Load conversation from Redis into memory."""
        if not self._redis_enabled:
            return []
        try:
            key  = self._redis_key(room_id)
            data = self._redis.get(key)
            if data:
                messages = json.loads(data)
                logger.info(
                    "ConversationMemory: restored %d messages "
                    "from Redis — room=%s ✅",
                    len(messages), room_id,
                )
                return messages
        except Exception as e:
            logger.warning("ConversationMemory Redis load failed: %s", e)
        return []

    # =========================================================================
    # PUBLIC — ADD MESSAGES
    # =========================================================================

    def add_user_message(
        self,
        room_id: str,
        content: str,
    ) -> None:
        if not content:
            return

        message = {
            "role":      "user",
            "content":   content.strip(),
            "timestamp": time.time(),
        }
        self._add_message(room_id, message)
        logger.info(
            "add_user_message — room=%s  content=%s",
            room_id, content[:60],
        )

    def add_ai_message(
        self,
        room_id: str,
        content: str,
    ) -> None:
        if not content:
            return

        message = {
            "role":      "assistant",
            "content":   content.strip(),
            "timestamp": time.time(),
        }
        self._add_message(room_id, message)
        logger.info(
            "add_ai_message — room=%s  content=%s",
            room_id, content[:60],
        )

    # =========================================================================
    # PUBLIC — GET HISTORY
    # =========================================================================

    def get_history(
        self,
        room_id: str,
        limit:   int = MAX_HISTORY,
    ) -> List[Dict]:
        """
        Get conversation history.
        
        Checks memory first, then Redis if memory empty.
        This handles server restarts automatically.
        """

        # ── Check memory first ────────────────────────────────────────────
        if room_id in self._conversations:
            history = self._conversations[room_id]
            return history[-limit:] if len(history) > limit else history

        # ── Not in memory — try Redis ──────────────────────────────────────
        restored = self._load_from_redis(room_id)
        if restored:
            self._conversations[room_id] = restored
            return restored[-limit:] if len(restored) > limit else restored

        return []

    # =========================================================================
    # PUBLIC — CLEAR
    # =========================================================================

    def clear(self, room_id: str) -> None:
        """Clear conversation history from memory and Redis."""

        count = 0
        if room_id in self._conversations:
            count = len(self._conversations[room_id])
            del self._conversations[room_id]

        if self._redis_enabled:
            try:
                self._redis.delete(self._redis_key(room_id))
            except Exception as e:
                logger.warning(
                    "ConversationMemory Redis clear failed: %s", e
                )

        logger.info(
            "clear — room=%s  removed %d message(s)",
            room_id, count,
        )

    # =========================================================================
    # PRIVATE — ADD MESSAGE
    # =========================================================================

    def _add_message(
        self,
        room_id: str,
        message: Dict,
    ) -> None:
        """
        Add message to memory and persist to Redis.
        
        If room not in memory, try to restore from Redis first.
        """

        # ── Restore from Redis if not in memory ───────────────────────────
        if room_id not in self._conversations:
            restored = self._load_from_redis(room_id)
            self._conversations[room_id] = restored if restored else []

        # ── Append message ─────────────────────────────────────────────────
        self._conversations[room_id].append(message)

        # ── Trim if needed ─────────────────────────────────────────────────
        if len(self._conversations[room_id]) > MAX_HISTORY:
            self._conversations[room_id] = (
                self._conversations[room_id][-MAX_HISTORY:]
            )
            logger.debug(
                "_add_message: trimmed to %d — room=%s",
                MAX_HISTORY, room_id,
            )

        # ── Persist to Redis ───────────────────────────────────────────────
        self._save_to_redis(room_id)


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

conversation_memory = ConversationMemory()