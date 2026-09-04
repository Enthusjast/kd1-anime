"""RAG 索引、检索、重排和运行时降级策略。"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kd1_anime.config import (
    DEFAULT_RAG_INDEX_PATH,
    LEGACY_RAG_INDEX_PATH,
    Settings,
    resolve_runtime_path,
    settings,
)
from kd1_anime.rag.chunker import (
    SourceChunk,
    chunk_file,
    iter_source_files,
    source_manifest_digest,
)
from kd1_anime.rag.clients import EmbeddingClient, RagClientError, RerankerClient
from kd1_anime.rag.models import (
    RagChunkRef,
    RagIndexInfo,
    RagReceipt,
    RagSearchResult,
    RagStatus,
    RetrievedChunk,
)
from kd1_anime.rag.store import RagIndex


@dataclass(frozen=True, slots=True)
class RagIndexBuildResult:
    """索引命令的摘要，不包含外部服务凭据。"""

    info: RagIndexInfo
    source_file_count: int
    chunk_count: int
    skipped_files: tuple[str, ...] = ()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _copy_private_file_if_missing(source: Path, destination: Path) -> bool:
    """原子复制旧缓存，避免迁移过程中留下半个 SQLite 文件。"""

    if destination.exists() or not source.is_file():
        return False
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.chmod(0o700)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        shutil.copyfile(source, temporary)
        temporary.chmod(0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # 不覆盖并发创建的新索引；临时文件与目标位于同一目录，因此硬链接
        # 的创建是原子的，失败时只需保留已经存在的目标。
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        if temporary is not None and temporary.exists():
            with suppress(OSError):
                temporary.unlink()
        return False
    return True


class RagService:
    """进程内共享的 RAG 服务；默认关闭且不会在构造时访问网络。"""

    def __init__(
        self, config: Settings | None = None, rag_semaphore: threading.Semaphore | None = None
    ):
        self.config = config or settings
        self.index_path = resolve_runtime_path(self.config.RAG_INDEX_PATH)
        self._migrate_legacy_index()
        self.embedding = EmbeddingClient(
            api_key=self.config.RAG_EMBEDDING_API_KEY,
            base_url=self.config.RAG_EMBEDDING_BASE_URL,
            model=self.config.RAG_EMBEDDING_MODEL,
            timeout=self.config.RAG_EMBEDDING_TIMEOUT,
        )
        self.reranker = RerankerClient(
            api_key=self.config.RAG_RERANK_API_KEY,
            base_url=self.config.RAG_RERANK_BASE_URL,
            model=self.config.RAG_RERANK_MODEL,
            timeout=self.config.RAG_RERANK_TIMEOUT,
        )
        self._semaphore = rag_semaphore or threading.BoundedSemaphore(
            max(1, self.config.RAG_PARALLEL_WORKERS)
        )
        self._cache_lock = threading.Lock()
        self._query_cache: dict[tuple[object, ...], RagSearchResult] = {}

    def _migrate_legacy_index(self) -> None:
        """把旧默认索引复制到新的应用目录；显式路径不受影响。"""

        if self.index_path != DEFAULT_RAG_INDEX_PATH.resolve():
            return
        _copy_private_file_if_missing(LEGACY_RAG_INDEX_PATH, self.index_path)

    @property
    def enabled(self) -> bool:
        return bool(self.config.RAG_ENABLED)

    @property
    def embedding_configured(self) -> bool:
        return bool(
            self.config.RAG_EMBEDDING_BASE_URL.strip() and self.config.RAG_EMBEDDING_MODEL.strip()
        )

    @property
    def reranker_configured(self) -> bool:
        return self.reranker.configured

    def _configured_source_roots(self) -> tuple[Path | None, Path | None]:
        """返回当前配置实际使用的源目录。"""

        docs = (
            resolve_runtime_path(self.config.RAG_DOCS_DIR)
            if self.config.RAG_DOCS_DIR is not None
            else None
        )
        examples = (
            resolve_runtime_path(self.config.RAG_EXAMPLES_DIR)
            if self.config.RAG_EXAMPLES_DIR is not None
            else None
        )
        return docs, examples

    def _source_digest_for_index(self, info: RagIndexInfo) -> str | None:
        """校验索引来源目录，并计算这些目录当前的源文件摘要。"""

        # 旧版/直接调用 RagIndex.build() 的索引没有记录源目录，无法可靠
        # 判断文件是否变化；保持兼容，不把这类已有索引误判为过期。
        if not info.source_docs_dir and not info.source_examples_dir:
            return None
        configured_docs, configured_examples = self._configured_source_roots()
        indexed_docs = (
            Path(info.source_docs_dir).expanduser().resolve()
            if info.source_docs_dir
            else configured_docs
        )
        indexed_examples = (
            Path(info.source_examples_dir).expanduser().resolve()
            if info.source_examples_dir
            else configured_examples
        )
        if (info.source_docs_dir or info.source_examples_dir) and (
            indexed_docs != configured_docs or indexed_examples != configured_examples
        ):
            raise ValueError("RAG 索引来源目录与当前配置不一致，请使用当前目录重新建立索引")
        return source_manifest_digest(indexed_docs, indexed_examples)

    def runtime_status(self) -> dict[str, Any]:
        """返回可直接展示的状态，不包含 API Key。"""

        info: RagIndexInfo | None = None
        index_error = ""
        index_stale = False
        current_source_sha256: str | None = None
        try:
            if self.index_path.is_file():
                info = RagIndex(self.index_path).verify_integrity()
                if (
                    self.embedding_configured
                    and info.embedding_model != self.config.RAG_EMBEDDING_MODEL
                ):
                    raise ValueError("RAG 索引使用了不同的 Embedding 模型，请重新建立索引")
                current_source_sha256 = self._source_digest_for_index(info)
                if (
                    current_source_sha256 is not None
                    and current_source_sha256 != info.source_sha256
                ):
                    index_stale = True
                    index_error = "知识库源文件已变化，请重新建立 RAG 索引"
        except (OSError, ValueError) as exc:
            index_error = str(exc)
            index_stale = "索引来源目录" in index_error
        if not self.enabled:
            status: RagStatus = "disabled"
        elif (
            not self.embedding_configured
            or info is None
            or not self.reranker_configured
            or bool(index_error)
        ):
            status = "degraded"
        else:
            status = "active"
        return {
            "status": status,
            "enabled": self.enabled,
            "index_path": str(self.index_path),
            "docs_dir": (
                str(resolve_runtime_path(self.config.RAG_DOCS_DIR))
                if self.config.RAG_DOCS_DIR is not None
                else ""
            ),
            "examples_dir": (
                str(resolve_runtime_path(self.config.RAG_EXAMPLES_DIR))
                if self.config.RAG_EXAMPLES_DIR is not None
                else ""
            ),
            "index": info.model_dump(mode="json") if info is not None else None,
            "index_error": index_error,
            "index_stale": index_stale,
            "current_source_sha256": current_source_sha256 or "",
            "embedding_model": self.config.RAG_EMBEDDING_MODEL,
            "embedding_configured": self.embedding_configured,
            "reranker_model": self.config.RAG_RERANK_MODEL,
            "reranker_configured": self.reranker_configured,
        }

    def require_index(self) -> None:
        """在启用 RAG 的生成入口前确认索引存在且没有过期。"""

        if not self.enabled:
            return
        status = self.runtime_status()
        if status["index"] is None:
            raise RagClientError("RAG 索引不存在，请先执行 `kd1-anime rag index` 构建索引")
        if status["index_error"]:
            raise RagClientError(str(status["index_error"]))

    def require_ready(self) -> None:
        """确认启用 RAG 的本地索引和两个外部服务配置完整。

        ``search`` 仍然保留运行时降级能力，便于已有任务在服务临时故障
        时记录 ``degraded``；但新生成入口不能在启动时就悄悄接受一个
        永远无法工作的 RAG 配置。网络连通性由 CLI 的 probe 负责。
        """

        if not self.enabled:
            return
        self.require_index()
        self.embedding.require()
        self.reranker.require()

    def _empty_result(
        self,
        query: str,
        stage: str,
        status: RagStatus,
        warning: str = "",
        *,
        code_sha256: str = "",
        inherited_elements_sha256: str = "",
    ) -> RagSearchResult:
        return RagSearchResult(
            context="",
            chunks=[],
            receipt=RagReceipt(
                stage=stage,
                query_sha256=_sha256(query),
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
                status=status,
                warning=warning,
            ),
        )

    def search(
        self,
        query: str,
        *,
        stage: str,
        source_kinds: set[str] | None = None,
        top_k: int | None = None,
        code_sha256: str = "",
        inherited_elements_sha256: str = "",
    ) -> RagSearchResult:
        """检索知识上下文；任何运行时服务错误都降级为无上下文。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("RAG query 必须是非空字符串")
        effective_top_k = self.config.RAG_TOP_K if top_k is None else top_k
        if effective_top_k < 1:
            raise ValueError("RAG top_k 必须大于 0")
        if not self.enabled:
            return self._empty_result(
                query,
                stage,
                "disabled",
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
            )
        if not self.embedding_configured:
            return self._empty_result(
                query,
                stage,
                "degraded",
                "Embedding 服务未配置，已跳过 RAG",
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
            )

        try:
            index = RagIndex(self.index_path)
            info = index.verify_integrity()
            if info.embedding_model != self.config.RAG_EMBEDDING_MODEL:
                raise ValueError("RAG 索引使用了不同的 Embedding 模型，请重新建立索引")
            current_source_sha256 = self._source_digest_for_index(info)
            if current_source_sha256 is not None and current_source_sha256 != info.source_sha256:
                raise ValueError("RAG 索引已过期：知识库源文件已变化，请重新建立索引")
            cache_key = (
                stage,
                query,
                tuple(sorted(source_kinds)) if source_kinds is not None else None,
                effective_top_k,
                self.config.RAG_RERANK_TOP_N,
                self.config.RAG_RERANK_MODEL,
                info.index_sha256,
                code_sha256,
                inherited_elements_sha256,
            )
            with self._cache_lock:
                cached = self._query_cache.get(cache_key)
            if cached is not None:
                return cached
            with self._semaphore:
                vector = self.embedding.embed([query])[0]
            candidates = index.search_verified(
                vector,
                top_k=effective_top_k,
                source_kinds=source_kinds,
                info=info,
            )
        except (OSError, ValueError, RagClientError) as exc:
            return self._empty_result(
                query,
                stage,
                "degraded",
                f"Embedding/索引检索失败，已跳过 RAG: {exc}",
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
            )

        warnings: list[str] = []
        ranked = candidates
        if self.reranker_configured and candidates:
            try:
                with self._semaphore:
                    reranked = self.reranker.rerank(
                        query,
                        [item.chunk.text for item in candidates],
                        top_n=self.config.RAG_RERANK_TOP_N,
                    )
                ranked = [
                    candidates[index].model_copy(update={"rerank_score": score})
                    for index, score in reranked
                ]
            except (ValueError, RagClientError) as exc:
                warnings.append(f"Reranker 不可用，已使用 Embedding 初排结果: {exc}")
        elif not self.reranker_configured:
            warnings.append("Reranker 服务未配置，已使用 Embedding 初排结果")

        context = self._format_context(ranked)
        refs = [
            RagChunkRef(
                chunk_id=item.chunk.chunk_id,
                content_sha256=item.chunk.content_sha256,
                source_path=item.chunk.source_path,
                score=item.score,
                rerank_score=item.rerank_score,
            )
            for item in ranked
        ]
        warning = "；".join(warnings)[:2_000]
        status: RagStatus = "degraded" if warnings else "active"
        result = RagSearchResult(
            context=context,
            chunks=ranked,
            receipt=RagReceipt(
                stage=stage,
                query_sha256=_sha256(query),
                index_sha256=info.index_sha256,
                code_sha256=code_sha256,
                inherited_elements_sha256=inherited_elements_sha256,
                status=status,
                chunks=refs,
                warning=warning,
            ),
        )
        if result.receipt.status == "active":
            with self._cache_lock:
                self._query_cache[cache_key] = result
                if len(self._query_cache) > 128:
                    self._query_cache.pop(next(iter(self._query_cache)))
        return result

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        blocks: list[str] = []
        used = 0
        limit = self.config.RAG_MAX_CONTEXT_CHARS
        for index, item in enumerate(chunks, start=1):
            block = (
                f'<reference index="{index}" source="{item.chunk.source_path}" '
                f'score="{item.rerank_score if item.rerank_score is not None else item.score:.4f}">\n'
                f"{item.chunk.text}\n"
                "</reference>"
            )
            if used + len(block) + 2 > limit:
                remaining = limit - used - 2
                if remaining > 80:
                    blocks.append(block[:remaining])
                break
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    def build_index(
        self,
        *,
        docs_dir: Path | None = None,
        examples_dir: Path | None = None,
        rebuild: bool = False,
    ) -> RagIndexBuildResult:
        """构建本地索引；未变化时复用，``rebuild`` 强制重新 Embedding。"""

        self.embedding.require()
        docs_root = docs_dir if docs_dir is not None else self.config.RAG_DOCS_DIR
        examples_root = examples_dir if examples_dir is not None else self.config.RAG_EXAMPLES_DIR
        docs_root = docs_root.expanduser().resolve() if docs_root is not None else None
        examples_root = examples_root.expanduser().resolve() if examples_root is not None else None
        sources = iter_source_files(docs_root, examples_root)
        if not sources:
            raise ValueError(
                "没有找到可索引的 .md/.rst/.py 文档或示例，请配置 RAG_DOCS_DIR/RAG_EXAMPLES_DIR"
            )
        if not rebuild:
            reusable = self._reusable_index_info(docs_root, examples_root)
            if reusable is not None:
                return RagIndexBuildResult(
                    info=reusable,
                    source_file_count=len(sources),
                    chunk_count=reusable.chunk_count,
                )
        chunks: list[SourceChunk] = []
        skipped: list[str] = []
        for path, source_kind in sources:
            try:
                root = docs_root if source_kind == "manim_doc" else examples_root
                display_path = (
                    f"{source_kind}/{path.relative_to(root).as_posix()}"
                    if root is not None
                    else path.name
                )
                chunks.extend(
                    chunk_file(
                        path,
                        source_kind,
                        chunk_size=self.config.RAG_CHUNK_SIZE,
                        overlap=self.config.RAG_CHUNK_OVERLAP,
                        display_path=display_path,
                    )
                )
            except (OSError, UnicodeError, ValueError) as exc:
                skipped.append(f"{path}: {exc}")
        if not chunks:
            raise ValueError("所有知识库源文件均无法产生有效分块")

        embeddings: list[list[float]] = []
        batch_size = self.config.RAG_EMBEDDING_BATCH_SIZE
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            with self._semaphore:
                embeddings.extend(self.embedding.embed([item.text for item in batch]))
        info = RagIndex.build(
            self.index_path,
            chunks,
            embeddings,
            embedding_model=self.config.RAG_EMBEDDING_MODEL,
            source_docs_dir=docs_root,
            source_examples_dir=examples_root,
        )
        return RagIndexBuildResult(
            info=info,
            source_file_count=len(sources) - len(skipped),
            chunk_count=len(chunks),
            skipped_files=tuple(skipped),
        )

    def _reusable_index_info(
        self,
        docs_root: Path | None,
        examples_root: Path | None,
    ) -> RagIndexInfo | None:
        """返回与当前源目录、Embedding 模型完全匹配的已有索引。"""

        try:
            info = RagIndex(self.index_path).verify_integrity()
            if info.embedding_model != self.config.RAG_EMBEDDING_MODEL:
                return None
            if not info.source_docs_dir and not info.source_examples_dir:
                # 没有源目录元数据的旧索引无法判断外部文件是否变化；只有
                # 当前调用也明确没有配置源目录时才可以安全复用。若当前
                # 配置有源目录，必须重建并写入新的来源身份。
                return info if docs_root is None and examples_root is None else None
            indexed_docs = (
                Path(info.source_docs_dir).expanduser().resolve()
                if info.source_docs_dir
                else docs_root
            )
            indexed_examples = (
                Path(info.source_examples_dir).expanduser().resolve()
                if info.source_examples_dir
                else examples_root
            )
            if indexed_docs != docs_root or indexed_examples != examples_root:
                return None
            current_source_sha256 = source_manifest_digest(docs_root, examples_root)
            if current_source_sha256 != info.source_sha256:
                return None
            return info
        except (OSError, ValueError):
            # 索引不存在、损坏或来源发生变化时走正常完整构建路径。
            return None

    def probe(self) -> None:
        """验证 Embedding 和 Reranker 的最小标准响应。"""

        self.embedding.require()
        with self._semaphore:
            vectors = self.embedding.embed(["kd1-anime RAG health check"])
        if not vectors or not vectors[0]:
            raise RagClientError("Embedding 健康检查返回空向量")
        self.reranker.require()
        with self._semaphore:
            results = self.reranker.rerank(
                "RAG health check",
                ["A short reference document."],
                top_n=1,
            )
        if not results:
            raise RagClientError("Reranker 健康检查返回空结果")
