from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule

_TEAM_COLORS = {"pro": "blue", "con": "red", "judge": "yellow"}


class TerminalDisplay:
    """Rich-based colored terminal display for debate output."""

    def __init__(self, console: Console | None = None):
        self._console = console or Console()

    def show_header(self, topic: str, pro_stance: str, con_stance: str) -> None:
        self._console.print()
        self._console.print(
            Panel(
                f"[bold]辩题：{topic}[/bold]\n"
                f"[blue]正方：{pro_stance}[/blue]    "
                f"[red]反方：{con_stance}[/red]",
                title="[bold]AI 辩论赛[/bold]",
                border_style="bright_white",
            )
        )

    def show_stage_banner(self, stage_name: str) -> None:
        self._console.print()
        self._console.print(Rule(f"[bold]{stage_name}[/bold]", style="bright_white"))
        self._console.print()

    def show_speech(
        self,
        name: str,
        team: str,
        content: str,
        time_used: float,
        time_limit: float,
        char_count: int,
        char_limit: int,
    ) -> None:
        color = _TEAM_COLORS.get(team, "white")
        minutes = int(time_used) // 60
        seconds = int(time_used) % 60
        limit_min = int(time_limit) // 60
        limit_sec = int(time_limit) % 60

        is_overtime = time_used > time_limit
        time_style = "red bold" if is_overtime else "green"
        time_mark = "!" if is_overtime else ""

        header = Text()
        header.append(f"[{name}]", style=f"bold {color}")
        header.append(f"  {char_count}/{char_limit}字", style="dim")
        header.append(
            f"  {minutes}:{seconds:02d}/{limit_min}:{limit_sec:02d}{time_mark}",
            style=time_style,
        )

        self._console.print(header)
        self._console.print(
            Panel(content, border_style=color, padding=(0, 1))
        )

    def show_overtime(
        self,
        name: str,
        team: str,
        excess_seconds: float,
        team_penalty: float,
        individual_penalty: float,
    ) -> None:
        self._console.print(
            f"  [magenta bold]超时 {excess_seconds:.0f}秒 — "
            f"{name} {individual_penalty}分, 队伍 {team_penalty}分[/magenta bold]"
        )

    def show_score(
        self,
        speaker_name: str,
        logic: int,
        persuasion: int,
        expression: int,
        teamwork: int,
        rule_compliance: int,
        total: float,
        comment: str,
    ) -> None:
        self._console.print(
            f"  [yellow][裁判][/yellow] "
            f"逻辑:[green]{logic}[/green] "
            f"说服力:[green]{persuasion}[/green] "
            f"表达:[green]{expression}[/green] "
            f"配合:[green]{teamwork}[/green] "
            f"规则:[green]{rule_compliance}[/green] "
            f"| [bold green]{total:.1f}[/bold green]分"
        )
        if comment:
            self._console.print(f"  [yellow]  \"{comment}\"[/yellow]")

    def show_violation(self, speaker_name: str, violation: str, penalty: float) -> None:
        self._console.print(
            f"  [magenta]违规: {speaker_name} — {violation} ({penalty}分)[/magenta]"
        )

    def show_scoreboard(self, pro_score: float, con_score: float) -> None:
        self._console.print()
        table = Table(title="当前比分", show_header=False, border_style="bright_white")
        table.add_column(justify="right")
        table.add_column(justify="center")
        table.add_column(justify="left")
        table.add_row(
            f"[bold blue]正方[/bold blue]",
            f"[bold green]{pro_score:.1f}[/bold green] : [bold green]{con_score:.1f}[/bold green]",
            f"[bold red]反方[/bold red]",
        )
        self._console.print(table)
