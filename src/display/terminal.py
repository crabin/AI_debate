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

    # Public API methods (called by CLI and stages)

    def header(self, title: str, pro_stance: str, con_stance: str) -> None:
        """Display debate header with topic and team stances."""
        self.show_header(title, pro_stance, con_stance)

    def participants(self, agents: dict) -> None:
        """Display debate participants list."""
        from rich.columns import Columns

        pro_debaters = [a for a in agents.values() if hasattr(a, 'team') and a.team == 'pro']
        con_debaters = [a for a in agents.values() if hasattr(a, 'team') and a.team == 'con']
        judge = agents.get('judge')

        def _agent_line(a) -> str:
            model = getattr(a, "model_name", "")
            model_tag = f" [{model}]" if model else ""
            return f"{a.name} ({a.position}辩){model_tag}"

        pro_names = [_agent_line(a) for a in sorted(pro_debaters, key=lambda x: x.position)]
        con_names = [_agent_line(a) for a in sorted(con_debaters, key=lambda x: x.position)]

        pro_panel = Panel(
            "\n".join(pro_names),
            title="[bold blue]正方[/bold blue]",
            border_style="blue",
        )
        con_panel = Panel(
            "\n".join(con_names),
            title="[bold red]反方[/bold red]",
            border_style="red",
        )

        self._console.print(Columns([pro_panel, con_panel]))
        if judge:
            judge_model = getattr(judge, "model_name", "")
            judge_model_tag = f" [{judge_model}]" if judge_model else ""
            self._console.print(f"[yellow]裁判：{judge.name}{judge_model_tag}[/yellow]")

    def final_results(self, results: dict) -> None:
        """Display final debate results."""
        self._console.print()
        self._console.print(Rule("[bold]辩论结果[/bold]", style="bright_white"))
        self.show_scoreboard(results["pro_score"], results["con_score"])

        winner_text = {
            "pro": "[bold blue]正方获胜！[/bold blue]",
            "con": "[bold red]反方获胜！[/bold red]",
            "tie": "[bold]平局！[/bold]",
        }.get(results["winner"], "")

        if winner_text:
            self._console.print(f"  {winner_text}  分差: {results['margin']:.1f}")

        best = results.get("best_debater")
        if best and best[0]:
            name, score = best
            self._console.print(f"  [yellow]最佳辩手：{name} ({score:.1f}分)[/yellow]")

        # Add verdict panel (new)
        self.verdict_panel(results)

        review = results.get("review")
        if review:
            self._console.print()
            # Convert review dict to formatted string
            if isinstance(review, dict):
                review_text = review.get("summary", "")
                highlights = review.get("highlights", [])
                suggestions = review.get("suggestions", [])

                parts = [review_text] if review_text else []
                if highlights:
                    parts.append("\n[yellow]亮点：[/yellow]")
                    parts.extend(f"  • {h}" for h in highlights)
                if suggestions:
                    parts.append("\n[yellow]建议：[/yellow]")
                    parts.extend(f"  • {s}" for s in suggestions)

                review_content = "\n".join(parts) if parts else "点评生成失败"
            else:
                review_content = str(review)

            self._console.print(Panel(review_content, title="[bold yellow]裁判总结[/bold yellow]", border_style="yellow"))

    def verdict_panel(self, results: dict) -> None:
        """Display the final verdict panel with topic conclusion and key moments."""
        topic_conclusion = results.get("topic_conclusion", "")
        winner_reason = results.get("winner_reason", "")
        key_moments = results.get("key_moments", [])
        best_debater_reason = results.get("best_debater_reason", "")

        if not any([topic_conclusion, winner_reason, key_moments]):
            return

        parts = []
        if winner_reason:
            parts.append(f"[bold]获胜原因：[/bold]{winner_reason}")
        if topic_conclusion:
            parts.append(f"\n[bold yellow]辩题结论：[/bold yellow]\n{topic_conclusion}")
        if key_moments:
            parts.append("\n[bold]关键时刻：[/bold]")
            parts.extend(f"  • {m}" for m in key_moments)
        if best_debater_reason:
            parts.append(f"\n[dim]{best_debater_reason}[/dim]")

        self._console.print(
            Panel("\n".join(parts), title="[bold yellow]裁判裁决[/bold yellow]", border_style="yellow")
        )

    def stage_start(self, name: str, description: str) -> None:
        """Display stage start banner."""
        self._console.print()
        self._console.print(Rule(f"[bold]{description}[/bold]", style="bright_white"))
        self._console.print()

    def stage_end(self, name: str) -> None:
        """Display stage end."""
        self._console.print()

    def speech(
        self,
        speaker: str,
        content: str,
        word_count: int,
        expected: int,
        time_used: float = 0.0,
        time_limit: float = 180.0,
    ) -> None:
        """Display a speech during debate.

        Args:
            speaker: Speaker name (e.g., "正方一辩")
            content: Speech content
            word_count: Character count of speech
            expected: Expected character limit
            time_used: Actual elapsed time in seconds
            time_limit: Time limit in seconds
        """
        # Extract team from speaker name if possible (assuming format like "正方一辩")
        team = "pro" if "正" in speaker else "con" if "反" in speaker else "judge"
        self.show_speech(
            name=speaker,
            team=team,
            content=content,
            time_used=time_used,
            time_limit=time_limit,
            char_count=word_count,
            char_limit=expected,
        )

    # Internal implementation methods (show_* prefix)

    def concurrent_speech_panels(
        self,
        pro_name: str,
        con_name: str,
        pro_buf: "io.StringIO",
        con_buf: "io.StringIO",
        pro_thread: "threading.Thread",
        con_thread: "threading.Thread",
        buf_lock: "threading.Lock",
        refresh_rate: float = 0.1,
    ) -> None:
        """Display two speakers' streaming output side-by-side using Rich Live.

        Polls both StringIO buffers until both threads finish, refreshing
        a two-panel table at ~10fps.

        Args:
            pro_name: Display name of the pro speaker
            con_name: Display name of the con speaker
            pro_buf: StringIO buffer written to by the pro thread
            con_buf: StringIO buffer written to by the con thread
            pro_thread: Pro speaker's thread (join signal)
            con_thread: Con speaker's thread (join signal)
            buf_lock: Lock shared with the writer threads
            refresh_rate: Seconds between Live refreshes
        """
        import io
        import threading
        import time as _time
        from rich.live import Live
        from rich.table import Table
        from rich.panel import Panel

        def _make_table() -> Table:
            with buf_lock:
                pro_text = pro_buf.getvalue()
                con_text = con_buf.getvalue()

            table = Table.grid(expand=True, padding=0)
            table.add_column(ratio=1)
            table.add_column(ratio=1)
            table.add_row(
                Panel(pro_text or "…", title=f"[bold blue]{pro_name}[/bold blue]", border_style="blue"),
                Panel(con_text or "…", title=f"[bold red]{con_name}[/bold red]", border_style="red"),
            )
            return table

        with Live(_make_table(), console=self._console, refresh_per_second=10) as live:
            while pro_thread.is_alive() or con_thread.is_alive():
                live.update(_make_table())
                _time.sleep(refresh_rate)
            # Final update with complete content
            live.update(_make_table())

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
            "[bold blue]正方[/bold blue]",
            f"[bold green]{pro_score:.1f}[/bold green] : [bold green]{con_score:.1f}[/bold green]",
            "[bold red]反方[/bold red]",
        )
        self._console.print(table)

    def show_judge_thinking(self, speaker_name: str) -> None:
        """Display judge thinking indicator."""
        header = Text()
        header.append(f"[评委]", style="bold yellow")
        header.append(f"  正在为 {speaker_name} 评分...", style="dim")
        self._console.print(header)

    def show_judge_score(
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
        """Display judge's score for a speaker."""
        from rich.table import Table

        table = Table(title=None, show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim")
        table.add_column()

        table.add_row("[yellow]评委[/yellow]", f"为 [bold]{speaker_name}[/bold] 评分")
        table.add_row("", "")
        table.add_row("  逻辑", f"[green]{logic}[/green]/10")
        table.add_row("  说服力", f"[green]{persuasion}[/green]/10")
        table.add_row("  表达", f"[green]{expression}[/green]/10")
        table.add_row("  配合", f"[green]{teamwork}[/green]/10")
        table.add_row("  规则", f"[green]{rule_compliance}[/green]/10")
        table.add_row("", "")
        table.add_row("  [bold]总分[/bold]", f"[bold green]{total:.1f}[/bold green]")

        self._console.print(table)

        if comment:
            self._console.print(f"  [yellow dim]\"{comment}\"[/yellow dim]")
        self._console.print()

    def show_judge_review_start(self) -> None:
        """Display judge review start indicator."""
        self._console.print()
        header = Text()
        header.append("[评委]", style="bold yellow")
        header.append("  正在生成总结...", style="dim")
        self._console.print(header)

    def show_judge_review_stream(self, text_chunk: str) -> None:
        """Display streaming judge review content."""
        self._console.print(text_chunk, end="")

    def show_judge_review_end(self) -> None:
        """End judge review display."""
        self._console.print()

    def speech_stream(
        self,
        speaker: str,
        content: str,
        word_count: int,
        expected: int,
        time_used: float = 0.0,
        time_limit: float = 180.0,
    ) -> None:
        """Display a speech with streaming character-by-character output.

        Args:
            speaker: Speaker name (e.g., "正方一辩")
            content: Speech content
            word_count: Character count of speech
            expected: Expected character limit
            time_used: Actual elapsed time in seconds
            time_limit: Time limit in seconds
        """
        import sys
        import time

        # Extract team from speaker name
        team = "pro" if "正" in speaker else "con" if "反" in speaker else "judge"
        color = _TEAM_COLORS.get(team, "white")

        # Show header
        minutes = int(time_used) // 60
        seconds = int(time_used) % 60
        limit_min = int(time_limit) // 60
        limit_sec = int(time_limit) % 60

        is_overtime = time_used > time_limit
        time_style = "red bold" if is_overtime else "green"
        time_mark = "!" if is_overtime else ""

        header = Text()
        header.append(f"[{speaker}]", style=f"bold {color}")
        header.append(f"  {word_count}/{expected}字", style="dim")
        header.append(
            f"  {minutes}:{seconds:02d}/{limit_min}:{limit_sec:02d}{time_mark}",
            style=time_style,
        )

        self._console.print(header)

        # Stream content character by character with inline printing
        # Using print directly to stdout for smooth streaming
        for char in content:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.008)  # Small delay for visual effect (~125 wpm)

        # Add newline after content
        sys.stdout.write("\n")
        sys.stdout.flush()
