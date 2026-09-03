from kd1_anime.agents.render_error_parser import extract_render_error


def test_extract_render_error_uses_last_traceback_and_source_context():
    log = """
Traceback (most recent call last):
  File "/tmp/scene_1.py", line 3, in construct
    self.play(Create(circle))
NameError: old_error

Traceback (most recent call last):
  File "/home/user/run/scene_1.py", line 8, in construct
    self.play(Write(title))
AttributeError: 'Text' object has no attribute 'write'
"""
    code = "\n".join(
        [
            "from manim import *",
            "",
            "",
            "",
            "",
            "",
            "        title = Text('x')",
            "        self.play(Write(title))",
        ]
    )

    evidence = extract_render_error(log, source_code=code)

    assert evidence.error_type == "AttributeError"
    assert "old_error" not in evidence.traceback
    assert evidence.file == "scene_1.py"
    assert evidence.line == 8
    assert "self.play(Write(title))" in evidence.code_line
    assert "8:         self.play(Write(title))" in evidence.source_context
    assert evidence.category == "python"
    assert len(evidence.fingerprint) == 16


def test_extract_render_error_handles_non_traceback_and_redacts_secrets():
    evidence = extract_render_error(
        "render failed with token=secret-key and TIMEOUT after 123 seconds",
        secrets=("secret-key",),
    )

    assert evidence.error_type == ""
    assert evidence.category == "timeout"
    assert "secret-key" not in evidence.traceback
    assert "<redacted>" in evidence.traceback
    assert evidence.message.endswith("seconds")


def test_extract_render_error_classifies_renderer_and_has_prompt_summary():
    evidence = extract_render_error(
        """
Traceback (most recent call last):
  File "scene_2.py", line 12, in construct
    self.play(Create(square))
AttributeError: 'OpenGLCamera' object has no attribute 'frame'
""",
        renderer="opengl",
    )

    assert evidence.category == "renderer"
    prompt = evidence.prompt_text()
    assert "OpenGLCamera" in prompt
    assert "scene_2.py:12" in prompt
