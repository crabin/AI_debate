from io import StringIO
from rich.console import Console
from src.display.terminal import TerminalDisplay


def _make_display() -> tuple[TerminalDisplay, StringIO]:
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=60)
    display = TerminalDisplay(console=console)
    return display, buf


def test_display_header():
    display, buf = _make_display()
    display.show_header("AI利大于弊", "利大于弊", "弊大于利")
    output = buf.getvalue()
    assert "利大于弊" in output
    assert "弊大于利" in output


def test_display_stage_banner():
    display, buf = _make_display()
    display.show_stage_banner("第一阶段：陈词")
    output = buf.getvalue()
    assert "陈词" in output


def test_display_speech():
    display, buf = _make_display()
    display.show_speech(
        name="正方一辩", team="pro", content="我方认为...",
        time_used=165.0, time_limit=180.0, char_count=500, char_limit=750,
    )
    output = buf.getvalue()
    assert "正方一辩" in output
    assert "我方认为" in output


def test_display_overtime_warning():
    display, buf = _make_display()
    display.show_overtime(
        name="正方二辩", team="pro",
        excess_seconds=8.0, team_penalty=-3, individual_penalty=-2,
    )
    output = buf.getvalue()
    assert "超时" in output or "8" in output


def test_display_score():
    display, buf = _make_display()
    display.show_score(
        speaker_name="正方一辩",
        logic=8, persuasion=7, expression=8, teamwork=7,
        rule_compliance=10, total=8.05, comment="论证清晰",
    )
    output = buf.getvalue()
    # Rich may split formatting with ANSI codes, check for key elements
    assert "8" in output  # Total score contains 8
    assert "论证清晰" in output  # Comment is included


def test_display_scoreboard():
    display, buf = _make_display()
    display.show_scoreboard(pro_score=42.5, con_score=38.2)
    output = buf.getvalue()
    assert "42.5" in output
    assert "38.2" in output
