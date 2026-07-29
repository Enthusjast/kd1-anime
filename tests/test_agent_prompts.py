from kd1_anime.agents.auto_fixer import AUTO_FIXER_SYSTEM_PROMPT, AutoFixerAgent
from kd1_anime.agents.coder import CODER_SYSTEM_PROMPT


def test_coder_requires_xelatex_xdv_and_ctex_template():
    assert 'tex_compiler="xelatex"' in CODER_SYSTEM_PROMPT
    assert 'output_format=".xdv"' in CODER_SYSTEM_PROMPT
    assert r"\usepackage{ctex}" in CODER_SYSTEM_PROMPT
    assert "config.tex_template" in CODER_SYSTEM_PROMPT
    assert "tex_template=tex_template" in CODER_SYSTEM_PROMPT


def test_auto_fixer_preserves_xelatex_invariant():
    assert 'tex_compiler="xelatex"' in AUTO_FIXER_SYSTEM_PROMPT
    assert "ctex" in AUTO_FIXER_SYSTEM_PROMPT
    assert AutoFixerAgent.is_infrastructure_error(
        "FileNotFoundError: No such file or directory: 'xelatex'"
    )
    assert not AutoFixerAgent.is_infrastructure_error(
        "FileNotFoundError: No such file or directory: 'pdflatex'"
    )
