"""把受信任根目录下的 Manim 文档和示例切成可检索文本。"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kd1_anime.rag.models import RagChunk

ALLOWED_SUFFIXES = frozenset({".md", ".rst", ".py"})
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "workspace",
        "runs",
    }
)
MAX_SOURCE_BYTES = 4 * 1024 * 1024
SourceKind = Literal["manim_doc", "example"]
_SENSITIVE_LINE = re.compile(
    r"(?:api[_ -]?key|access[_ -]?token|secret|password|private\s+key|bearer\s+[A-Za-z0-9._-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SourceChunk:
    path: Path
    source_kind: SourceKind
    source_sha256: str
    ordinal: int
    text: str
    metadata: dict[str, str]
    display_path: str = ""


def _safe_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    root = path.expanduser().resolve()
    return root if root.is_dir() else None


def iter_source_files(
    docs_dir: Path | None,
    examples_dir: Path | None,
) -> list[tuple[Path, SourceKind]]:
    """列出允许索引的文件，拒绝运行目录和隐藏构建目录。"""

    result: list[tuple[Path, SourceKind]] = []
    seen: set[Path] = set()
    for root, source_kind in (
        (_safe_root(docs_dir), "manim_doc"),
        (_safe_root(examples_dir), "example"),
    ):
        if root is None:
            continue
        for path in sorted(root.rglob("*")):
            try:
                resolved = path.resolve(strict=True)
                relative = resolved.relative_to(root)
            except (OSError, ValueError):
                # 不索引指向知识库根目录之外的符号链接。
                continue
            if (
                not resolved.is_file()
                or path.suffix.lower() not in ALLOWED_SUFFIXES
                or resolved in seen
                # 必须检查相对根目录的路径。若检查绝对路径，用户把知识库
                # 放在 ``~/workspace/docs`` 下时，会因为父目录名 workspace
                # 而把整个合法知识库误过滤掉。
                or any(part in EXCLUDED_PARTS for part in relative.parts)
            ):
                continue
            try:
                if resolved.stat().st_size > MAX_SOURCE_BYTES:
                    continue
            except OSError:
                continue
            seen.add(resolved)
            result.append((resolved, source_kind))
    return result


def source_manifest_digest(
    docs_dir: Path | None,
    examples_dir: Path | None,
) -> str | None:
    """计算当前知识库源文件的摘要，用于发现索引过期。

    摘要格式与 ``RagIndex`` 写入索引时使用的 source_sha256 集合一致。
    没有配置任何源目录时返回 ``None``；目录已配置但为空时仍返回摘要，
    这样删除全部源文件也会被识别为索引失效。
    """

    roots = {
        "manim_doc": _safe_root(docs_dir),
        "example": _safe_root(examples_dir),
    }
    if docs_dir is None and examples_dir is None:
        return None

    values: list[tuple[str, str]] = []
    for path, source_kind in iter_source_files(docs_dir, examples_dir):
        root = roots[source_kind]
        if root is None:
            continue
        try:
            text, source_sha256 = _read_source(path)
        except (OSError, UnicodeError, ValueError):
            # 与 build_index 一致：无法读取的文件不参与当前可检索源摘要。
            continue
        # 空文件或只包含被过滤凭据行的文件不会产生 chunk，不能让它们
        # 造成一个永远无法与已构建索引匹配的摘要。
        if not text:
            continue
        display_path = f"{source_kind}/{path.relative_to(root).as_posix()}"
        values.append((display_path, source_sha256))
    payload = json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_source(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError(f"知识库源文件过大: {path}")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")
    # 删除疑似凭据所在的整行；知识库宁可少一行，也不把密钥送到外部服务。
    safe_lines = [line for line in text.splitlines() if not _SENSITIVE_LINE.search(line)]
    return "\n".join(safe_lines).strip(), source_sha256


def _markdown_segments(text: str) -> list[str]:
    segments = [segment.strip() for segment in re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)]
    return [segment for segment in segments if segment]


def _rst_segments(text: str) -> list[str]:
    """按 reStructuredText 的标题下划线切分文档。"""

    heading = re.compile(r"(?m)^[^\s].*\n[=\-`:\.'\"~^_*+#]{3,}\s*$")
    starts = [match.start() for match in heading.finditer(text)]
    if not starts:
        return [text]
    if starts[0] > 0:
        starts.insert(0, 0)
    boundaries = [*starts, len(text)]
    return [text[boundaries[index] : boundaries[index + 1]].strip() for index in range(len(starts))]


def _python_segments(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [text]
    segments: list[str] = []
    for node in tree.body:
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        segment = "\n".join(lines[start:end]).strip()
        if segment:
            segments.append(segment)
    return segments or [text]


def _split_long_segment(segment: str, chunk_size: int, overlap: int) -> list[str]:
    if len(segment) <= chunk_size:
        return [segment]
    step = max(1, chunk_size - overlap)
    return [
        segment[start : start + chunk_size].strip()
        for start in range(0, len(segment), step)
        if segment[start : start + chunk_size].strip()
    ]


def _pack_segments(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    expanded: list[str] = []
    for segment in segments:
        expanded.extend(_split_long_segment(segment, chunk_size, overlap))
    packed: list[str] = []
    current = ""
    for segment in expanded:
        candidate = f"{current}\n\n{segment}" if current else segment
        if current and len(candidate) > chunk_size:
            packed.append(current.strip())
            current = segment
        else:
            current = candidate
    if current.strip():
        packed.append(current.strip())
    return packed


def chunk_file(
    path: Path,
    source_kind: SourceKind,
    *,
    chunk_size: int = 1_800,
    overlap: int = 200,
    display_path: str | None = None,
) -> list[SourceChunk]:
    """切分单个源文件；调用方负责处理读取失败。"""

    if source_kind not in {"manim_doc", "example"}:
        raise ValueError(f"不支持的知识库来源类型: {source_kind}")
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须 >=100，且 overlap 必须满足 0 <= overlap < chunk_size")
    text, source_sha256 = _read_source(path)
    if not text:
        return []
    suffix = path.suffix.lower()
    if suffix == ".md":
        segments = _markdown_segments(text)
    elif suffix == ".rst":
        segments = _rst_segments(text)
    else:
        segments = _python_segments(text)
    contents = _pack_segments(segments, chunk_size, overlap)
    result: list[SourceChunk] = []
    for ordinal, content in enumerate(contents):
        result.append(
            SourceChunk(
                path=path,
                source_kind=source_kind,
                source_sha256=source_sha256,
                ordinal=ordinal,
                text=content,
                metadata={"suffix": path.suffix.lower()},
                display_path=display_path or path.name,
            )
        )
    return result


def to_rag_chunk(item: SourceChunk) -> RagChunk:
    content_sha256 = hashlib.sha256(item.text.encode("utf-8")).hexdigest()
    chunk_id = hashlib.sha256(
        f"{item.source_sha256}:{item.source_kind}:{item.ordinal}:{content_sha256}".encode()
    ).hexdigest()
    return RagChunk(
        chunk_id=chunk_id,
        source_path=item.display_path or item.path.name,
        source_kind=item.source_kind,
        source_sha256=item.source_sha256,
        content_sha256=content_sha256,
        ordinal=item.ordinal,
        text=item.text,
        metadata=item.metadata,
    )
