"""可选的本地知识检索层。"""

from kd1_anime.rag.clients import EmbeddingClient, RagClientError, RerankerClient
from kd1_anime.rag.models import (
    RagChunk,
    RagChunkRef,
    RagIndexInfo,
    RagReceipt,
    RagRuntimeProfile,
    RagSearchResult,
    RagStatus,
    RetrievedChunk,
)
from kd1_anime.rag.recipes import RecipeRecord, RecipeStore, anonymize_code
from kd1_anime.rag.service import RagService

__all__ = [
    "EmbeddingClient",
    "RagChunk",
    "RagChunkRef",
    "RagClientError",
    "RagIndexInfo",
    "RagReceipt",
    "RagRuntimeProfile",
    "RagSearchResult",
    "RagService",
    "RagStatus",
    "RecipeRecord",
    "RecipeStore",
    "RerankerClient",
    "RetrievedChunk",
    "anonymize_code",
]
