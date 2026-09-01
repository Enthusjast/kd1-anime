"""LLM 上下文预算和区块裁剪工具。"""

from __future__ import annotations

from dataclasses import dataclass


class PromptBudgetError(ValueError):
    """必需的 Prompt 区块超过总预算。"""


@dataclass(frozen=True, slots=True)
class PromptSection:
    """一段可按优先级裁剪的 Prompt 内容。"""

    name: str
    content: str
    required: bool = False
    priority: int = 0
    max_chars: int | None = None
    # 结构化数据（如 RAG JSON 块或元素清单）不能从中间截断；空间不足时
    # 整段省略，不能把破损 JSON 交给模型。
    atomic: bool = False


class PromptContextBuilder:
    """在不破坏代码/合同区块的前提下构造有界上下文。"""

    def __init__(self, max_chars: int) -> None:
        if isinstance(max_chars, bool) or max_chars < 1:
            raise ValueError("Prompt 上下文预算必须是正整数")
        self.max_chars = max_chars

    @staticmethod
    def _clip(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        if limit <= 80:
            return content[:limit]
        marker = "\n...[该区块因上下文预算被裁剪]...\n"
        available = max(1, limit - len(marker))
        head = (available + 1) // 2
        tail = available - head
        return content[:head] + marker + (content[-tail:] if tail else "")

    @staticmethod
    def _render(section: PromptSection, content: str | None = None) -> str:
        value = section.content if content is None else content
        if not value.strip():
            return ""
        return f"### {section.name}\n{value.strip()}"

    def build(self, sections: list[PromptSection]) -> str:
        normalized: list[PromptSection] = []
        names: set[str] = set()
        for section in sections:
            name = str(section.name).strip()
            content = str(section.content)
            if not name:
                raise ValueError("Prompt 区块名称不能为空")
            if name in names:
                raise ValueError(f"Prompt 区块名称重复: {name}")
            names.add(name)
            if content.strip():
                normalized.append(
                    PromptSection(
                        name=name,
                        content=content,
                        required=section.required,
                        priority=section.priority,
                        max_chars=section.max_chars,
                        atomic=section.atomic,
                    )
                )
        rendered: list[str] = []
        for section in normalized:
            if section.required:
                if section.max_chars is not None and len(section.content) > section.max_chars:
                    raise PromptBudgetError(
                        f"必需 Prompt 区块 {section.name} 超过区块预算 {section.max_chars} 字符"
                    )
                content = section.content
            else:
                if section.atomic and section.max_chars is not None:
                    content = section.content if len(section.content) <= section.max_chars else ""
                else:
                    content = (
                        self._clip(section.content, section.max_chars)
                        if section.max_chars is not None
                        else section.content
                    )
            rendered.append(self._render(section, content))
        separator_length = 2 * max(0, len(rendered) - 1)
        if sum(len(item) for item in rendered) + separator_length <= self.max_chars:
            return "\n\n".join(rendered)

        required = [
            self._render(section, section.content) for section in normalized if section.required
        ]
        required_length = sum(len(item) for item in required) + 2 * max(0, len(required) - 1)
        if required_length > self.max_chars:
            names = ", ".join(section.name for section in normalized if section.required)
            raise PromptBudgetError(
                f"必需 Prompt 区块超过上下文预算 {self.max_chars} 字符: {names}"
            )

        optional = [section for section in normalized if not section.required]
        remaining = self.max_chars - required_length
        # 低优先级区块先让出空间；同一优先级按原始顺序处理。
        selected: dict[str, str] = {}
        for section in sorted(optional, key=lambda item: item.priority, reverse=True):
            if remaining <= 2:
                break
            rendered_section = self._render(section)
            available = remaining - 2
            if section.atomic:
                if len(rendered_section) > available:
                    continue
                selected[section.name] = rendered_section
                remaining -= len(rendered_section) + 2
                continue
            if section.max_chars is not None:
                available = min(available, section.max_chars + len(f"### {section.name}\n"))
            clipped = self._clip(rendered_section, available)
            if len(clipped) < len(f"### {section.name}\n") + 1:
                continue
            selected[section.name] = clipped
            remaining -= len(clipped) + 2

        result: list[str] = []
        for section in normalized:
            if section.required:
                value = self._render(section, section.content)
            else:
                value = selected.get(section.name, "")
            if value:
                result.append(value)
        final = "\n\n".join(result)
        if len(final) > self.max_chars:
            raise PromptBudgetError(f"Prompt 区块裁剪后仍超过上下文预算 {self.max_chars} 字符")
        return final


def build_bounded_prompt(
    sections: list[PromptSection],
    *,
    max_chars: int,
) -> str:
    """便捷函数，供 Agent 构造最终 user message。"""

    return PromptContextBuilder(max_chars).build(sections)
