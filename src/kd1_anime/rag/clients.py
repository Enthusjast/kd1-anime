"""Embedding 和 reranker 的独立 HTTP 客户端。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from kd1_anime.security import redact_text


class RagClientError(RuntimeError):
    """RAG 外部服务不可用或返回结构不合法。"""


MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _endpoint(base_url: str, suffix: str) -> str:
    parsed = urlsplit(base_url.strip())
    path = parsed.path.rstrip("/")
    if not path.endswith(f"/{suffix}"):
        path = f"{path}/{suffix}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _safe_endpoint(url: str) -> str:
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or "<invalid-host>"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        return "<invalid-endpoint>"


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    *,
    trust_env: bool = True,
) -> Any:
    try:
        import httpx

        # 使用 streaming + 上限读取，而不是先访问 response.content；后者
        # 会在检查大小前把异常大的响应完整放入内存。
        with (
            httpx.Client(timeout=timeout, trust_env=trust_env) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise RagClientError("RAG 服务返回了非法 Content-Length") from exc
                if declared_length > MAX_RESPONSE_BYTES:
                    raise RagClientError("RAG 服务响应超过大小限制")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise RagClientError("RAG 服务响应超过大小限制")
                chunks.append(chunk)
            content = b"".join(chunks)
            if len(content) > MAX_RESPONSE_BYTES:
                raise RagClientError("RAG 服务响应超过大小限制")
            try:
                return json.loads(content)
            except (TypeError, json.JSONDecodeError) as exc:
                raise RagClientError("RAG 服务返回的 JSON 无法解析") from exc
    except Exception as exc:
        if isinstance(exc, RagClientError):
            raise
        detail = redact_text(str(exc).strip() or type(exc).__name__, headers.values())
        raise RagClientError(f"RAG 服务请求失败（{_safe_endpoint(url)}）: {detail}") from exc


class EmbeddingClient:
    """调用 OpenAI-compatible Embeddings 接口。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        trust_env: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.trust_env = trust_env

    def require(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise RagClientError("Embedding 服务配置不完整，需要 RAG_EMBEDDING_BASE_URL 和 MODEL")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("Embedding 输入必须是非空字符串序列")
        self.require()
        payload = {
            "model": self.model,
            "input": values,
            "encoding_format": "float",
        }
        data = _post_json(
            _endpoint(self.base_url, "embeddings"),
            _headers(self.api_key),
            payload,
            self.timeout,
            trust_env=self.trust_env,
        )
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise RagClientError("Embedding 响应缺少 data 数组")
        records = data["data"]
        if len(records) != len(values):
            raise RagClientError(f"Embedding 返回数量不一致: {len(records)} != {len(values)}")
        ordered: list[list[float] | None] = [None] * len(values)
        for fallback_index, record in enumerate(records):
            if not isinstance(record, dict) or not isinstance(record.get("embedding"), list):
                raise RagClientError("Embedding 响应中的 embedding 不是数组")
            index = record.get("index", fallback_index)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(values)
            ):
                raise RagClientError(f"Embedding 响应包含非法 index: {index!r}")
            if ordered[index] is not None:
                raise RagClientError(f"Embedding 响应包含重复 index: {index}")
            vector = record["embedding"]
            if not vector or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                for item in vector
            ):
                raise RagClientError("Embedding 向量为空或包含非法数值")
            ordered[index] = [float(item) for item in vector]
        if any(vector is None for vector in ordered):
            raise RagClientError("Embedding 响应缺少部分 index")
        vectors = [vector for vector in ordered if vector is not None]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise RagClientError("同一批 Embedding 的向量维度不一致")
        return vectors


class RerankerClient:
    """调用 Cohere-compatible rerank 接口。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        trust_env: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.trust_env = trust_env

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip() and self.model.strip())

    def require(self) -> None:
        if not self.configured:
            raise RagClientError("Reranker 服务配置不完整，需要 RAG_RERANK_BASE_URL 和 MODEL")

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> list[tuple[int, float]]:
        values = list(documents)
        if not query.strip() or not values:
            return []
        if top_n < 1:
            raise ValueError("rerank top_n 必须大于 0")
        self.require()
        payload = {
            "model": self.model,
            "query": query,
            "documents": values,
            "top_n": min(top_n, len(values)),
            "return_documents": False,
        }
        data = _post_json(
            _endpoint(self.base_url, "rerank"),
            _headers(self.api_key),
            payload,
            self.timeout,
            trust_env=self.trust_env,
        )
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RagClientError("Reranker 响应缺少 results 数组")
        result: list[tuple[int, float]] = []
        seen: set[int] = set()
        for record in data["results"]:
            if not isinstance(record, dict):
                raise RagClientError("Reranker results 中存在非法项目")
            index = record.get("index")
            score = record.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(values)
                or index in seen
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise RagClientError(f"Reranker 响应包含非法结果: {record!r}")
            seen.add(index)
            result.append((index, float(score)))
        if not result:
            raise RagClientError("Reranker 响应为空")
        result.sort(key=lambda item: item[1], reverse=True)
        return result[:top_n]
