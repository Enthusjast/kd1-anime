from io import StringIO

from rich.console import Console

import kd1_anime.tui as tui_module
from kd1_anime.logo import ASCII_LOGO, print_logo
from kd1_anime.tui import ChatSession


def test_logo_is_a_terminal_safe_ansi_representation():
    output = StringIO()
    print_logo(output)

    assert output.getvalue() == ASCII_LOGO
    assert "@" in ASCII_LOGO
    assert "#" in ASCII_LOGO
    assert "\x1b" not in ASCII_LOGO


def test_tui_completion_displays_logo_for_successful_dry_run(monkeypatch):
    output = StringIO()
    monkeypatch.setattr(
        tui_module,
        "console",
        Console(file=output, force_terminal=False, color_system=None),
    )

    ChatSession._show_completion(None, dry_run=True)

    rendered = output.getvalue()
    assert "Dry-run 已完成" in rendered
    assert "@" in rendered
