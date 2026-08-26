"""RAG 的内部数据模型；所有模型都不包含密钥或完整请求内容。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RagStatus = Literal["disabled", "active", "degraded"]


class RagChunk(BaseModel):
    """知识库中的一个文本分块。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str = Field(min_length=1, max_length=4_000)
    source_kind: Literal["manim_doc", "example"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=30)


class RetrievedChunk(BaseModel):
    """一次检索返回的分块及其排序分数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk: RagChunk
    score: float = Field(ge=-1.0, le=1.0)
    rerank_score: float | None = None


class RagChunkRef(BaseModel):
    """写入收据的最小分块身份信息。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str = Field(min_length=1, max_length=4_000)
    score: float
    rerank_score: float | None = None


class RagReceipt(BaseModel):
    """把注入 Agent 的知识上下文绑定到索引和分块哈希。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1, max_length=100)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    code_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    inherited_elements_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    status: RagStatus
    chunks: list[RagChunkRef] = Field(default_factory=list, max_length=100)
    warning: str = Field(default="", max_length=2_000)


class RagRuntimeProfile(BaseModel):
    """写入运行清单的非敏感 RAG 配置快照。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    status: RagStatus = "disabled"
    index_sha256: str = Field(default="", pattern=r"^(?:[0-9a-f]{64})?$")
    embedding_model: str = Field(default="", max_length=500)
    reranker_model: str = Field(default="", max_length=500)
    top_k: int = Field(default=8, ge=1, le=100)
    rerank_top_n: int = Field(default=4, ge=1, le=100)
    max_context_chars: int = Field(default=12_000, ge=500, le=100_000)
    evaluator_version: Literal["1"] = "1"


class RagSearchResult(BaseModel):
    """检索文本与可审计收据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: str = Field(default="", max_length=50_000)
    chunks: list[RetrievedChunk] = Field(default_factory=list, max_length=100)
    receipt: RagReceipt


class RagIndexInfo(BaseModel):
    """本地索引的非敏感元数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    index_path: str = Field(min_length=1, max_length=4_000)
    index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: str = Field(min_length=1, max_length=500)
    embedding_dimension: int = Field(gt=0)
    chunk_count: int = Field(ge=0)
