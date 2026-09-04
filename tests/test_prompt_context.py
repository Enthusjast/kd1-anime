"""LLM Prompt 上下文预算测试。"""

import pytest

from kd1_anime.agents.prompt_context import (
    PromptBudgetError,
    PromptSection,
    build_bounded_prompt,
)


def test_prompt_builder_keeps_required_sections_and_trims_low_priority_context():
    prompt = build_bounded_prompt(
        [
            PromptSection("contract", "CONTRACT", required=True, priority=100),
            PromptSection("rag", "R" * 500, priority=1),
            PromptSection("notes", "N" * 500, priority=2),
        ],
        max_chars=160,
    )

    assert "CONTRACT" in prompt
    assert len(prompt) <= 160
    assert "### contract" in prompt


def test_prompt_builder_rejects_required_section_overflow():
    with pytest.raises(PromptBudgetError, match="必需 Prompt 区块"):
        build_bounded_prompt(
            [PromptSection("code", "x" * 101, required=True, max_chars=100)],
            max_chars=1_000,
        )


def test_prompt_builder_does_not_truncate_required_code():
    content = "from manim import *\n" + "x" * 90

    prompt = build_bounded_prompt(
        [PromptSection("code", content, required=True)],
        max_chars=len(content) + 20,
    )

    assert content in prompt


def test_prompt_builder_rejects_duplicate_or_empty_section_names():
    with pytest.raises(ValueError, match="区块名称重复"):
        build_bounded_prompt(
            [PromptSection("contract", "a"), PromptSection("contract", "b")],
            max_chars=100,
        )
    with pytest.raises(ValueError, match="区块名称不能为空"):
        build_bounded_prompt([PromptSection("", "a")], max_chars=100)
