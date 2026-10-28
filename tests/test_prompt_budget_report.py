import pytest

from kd1_anime.agents.prompt_context import (
    PromptBudgetError,
    PromptContextBuilder,
    PromptSection,
    estimate_tokens,
)


def test_context_report_counts_cjk_system_and_output_reserve():
    sections = [
        PromptSection("required", "公式：x² + y² = z²", required=True),
        PromptSection("rag", "中文资料" * 200),
    ]
    report = PromptContextBuilder(4000).build(
        sections,
        system_prompt="系统安全合同",
        output_reserve=100,
        safety_margin=10,
        total_budget=200,
    )
    assert "公式：x² + y² = z²" in report.text
    assert report.estimated_prompt_tokens == estimate_tokens("系统安全合同") + estimate_tokens(
        report.text
    )
    assert report.estimated_prompt_tokens + report.output_reserve + report.safety_margin <= 200
    assert "rag" in report.clipped_sections or "rag" in report.omitted_sections


def test_required_contract_is_never_truncated_for_token_budget():
    with pytest.raises(PromptBudgetError):
        PromptContextBuilder(1000).build(
            [PromptSection("required", "数学证明" * 100, required=True)],
            total_budget=30,
            output_reserve=10,
        )


def test_atomic_optional_section_is_omitted_without_partial_json():
    source = '{"references": [' + '"公式",' * 50 + '"last"]}'
    report = PromptContextBuilder(1000).build(
        [
            PromptSection("contract", "required", required=True),
            PromptSection("rag", source, atomic=True),
        ],
        total_budget=60,
    )
    assert "rag" in report.omitted_sections and source not in report.text
