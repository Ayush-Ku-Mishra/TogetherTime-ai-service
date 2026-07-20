# ai-service/models/ai_models.py
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any


class SubtitleItem(BaseModel):
    text: str = ""
    timestamp: Optional[float] = 0


class SceneSummaryItem(BaseModel):
    timestamp: Optional[float] = 0
    summary: str = ""


class ConversationItem(BaseModel):
    question: Optional[str] = ""
    answer: Optional[str] = ""


class AIRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    roomId:               str
    userId:               Optional[str]   = ""
    movieTitle:           Optional[str]   = "Unknown Content"
    movieId:              Optional[str]   = None
    currentTimestamp:     Optional[float] = 0
    recentSubtitles:      Optional[List[SubtitleItem]]    = []
    recentSceneSummaries: Optional[List[SceneSummaryItem]] = []
    conversationHistory:  Optional[List[ConversationItem]] = []
    userQuestion:         str
    userLanguage:         Optional[str]  = "en"
    spoilerProtection:    Optional[bool] = True
    platform:             Optional[str]  = "youtube"

    @field_validator("userQuestion")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or v.strip() == "" or v.strip() == "undefined":
            raise ValueError("userQuestion cannot be empty or undefined")
        return v.strip()

    @field_validator("movieTitle")
    @classmethod
    def fix_movie_title(cls, v):
        if not v or v.strip() in ["", "undefined", "Movie", "null"]:
            return "the content being watched"
        return v.strip()


class AIResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    answer:     str
    model_used: str  = "unknown"
    success:    bool = True


class RecommendRequest(BaseModel):
    preferences:  Dict[str, Any] = {}
    groupSize:    int             = 1
    currentMovie: Optional[str]  = None


class ProcessVideoRequest(BaseModel):
    roomId:     str
    videoId:    str
    movieTitle: Optional[str] = ""
    language:   Optional[str] = None
    platform:   Optional[str] = "youtube"


class SessionSummaryRequest(BaseModel):
    roomId:           str
    movieTitle:       str
    fullSceneHistory: List[Dict] = []
    chatHighlights:   List[Dict] = []
    watchDuration:    int        = 0
    participantCount: int        = 1


class TranslateRequest(BaseModel):
    text:           str
    targetLanguage: str           = "en"
    context:        Optional[str] = ""