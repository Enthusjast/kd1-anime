from io import StringIO

from rich.console import Console

import kd1_anime.tui as tui_module
from kd1_anime.logo import (
    ASCII_LOGO,
    ASCII_LOGO_COLORS,
    BRAND_SUBTITLE,
    BRAND_TITLE,
    COLORED_ASCII_BRAND,
    COLORED_ASCII_LOGO,
    print_logo,
)
from kd1_anime.tui import ChatSession


def test_logo_is_a_terminal_safe_ansi_representation():
    output = StringIO()
    print_logo(output)

    assert output.getvalue() == COLORED_ASCII_LOGO
    assert "@" in ASCII_LOGO
    assert "#" in ASCII_LOGO
    assert set("@#+.").issubset(ASCII_LOGO_COLORS)
    assert "\x1b[38;2;0;113;188m" in COLORED_ASCII_LOGO
    assert "\x1b[38;2;117;76;36m" in COLORED_ASCII_LOGO


def test_colored_brand_contains_right_side_identity():
    assert BRAND_TITLE in COLORED_ASCII_BRAND
    assert BRAND_SUBTITLE in COLORED_ASCII_BRAND


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
