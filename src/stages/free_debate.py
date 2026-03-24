"""Free debate stage for AI Debate System."""

import io
import time
import threading
import logging
from src.stages.base import BaseStage
from src.engine.message_pool import Message
from src.engine.timer import Timer

logger = logging.getLogger(__name__)


class FreeDebateStage(BaseStage):
    """Free debate stage (自由辩论阶段).

    Alternating turns between teams:
    - Pro starts, then teams alternate
    - Each team has 4 minutes total (240 seconds)
    - Each debater must speak at least once
    - Cannot speak twice in a row on same team
    """

    # Each team's time in seconds
    _TEAM_TIME = 240  # 4 minutes

    # Characters per minute for Chinese speech rate
    _CHARS_PER_MINUTE = 250

    # Minimum turns per debater
    _REQUIRED_SPEAKERS = ["pro_1", "pro_2", "pro_3", "pro_4",
                          "con_1", "con_2", "con_3", "con_4"]

    # Team membership
    _TEAMS = {
        "pro": ["pro_1", "pro_2", "pro_3", "pro_4"],
        "con": ["con_1", "con_2", "con_3", "con_4"],
    }

    def __init__(self, display, timer: Timer | None = None) -> None:
        """Initialize free debate stage.

        Args:
            display: TerminalDisplay instance for output
            timer: Optional Timer instance (creates default if None)
        """
        super().__init__(
            name="free_debate",
            description="自由辩论阶段：正反方各4分钟，交替发言",
            display=display,
        )
        self._timer = timer or Timer(
            total_seconds=self._TEAM_TIME,
            chars_per_minute=self._CHARS_PER_MINUTE,
        )

    @classmethod
    def create(cls, display, timer: Timer | None = None) -> "FreeDebateStage":
        """Factory method to create a FreeDebateStage.

        Args:
            display: TerminalDisplay instance for output
            timer: Optional Timer instance

        Returns:
            Configured FreeDebateStage instance
        """
        return cls(display=display, timer=timer)

    def _get_next_speaker(
        self,
        agents: dict,
        current_team: str,
        speak_counts: dict,
        last_speaker: str | None,
    ) -> tuple[str | None, str]:
        """Select next speaker for free debate.

        Args:
            agents: Available agents
            current_team: Team that should speak next
            speak_counts: Count of speeches per agent
            last_speaker: Previous speaker ID

        Returns:
            Tuple of (next_speaker_id, next_team)
        """
        available = [
            agent_id for agent_id in self._TEAMS[current_team]
            if agent_id in agents and agent_id != last_speaker
        ]

        if not available:
            # Fallback: allow same speaker if only option
            available = [
                agent_id for agent_id in self._TEAMS[current_team]
                if agent_id in agents
            ]

        if not available:
            return None, ("con" if current_team == "pro" else "pro")

        # Prefer speakers who haven't spoken yet
        unspeaked = [
            agent_id for agent_id in available
            if speak_counts.get(agent_id, 0) == 0
        ]

        if unspeaked:
            next_speaker = unspeaked[0]
        else:
            # Select speaker with fewest speeches
            next_speaker = min(
                available,
                key=lambda x: speak_counts.get(x, 0),
            )

        next_team = "con" if current_team == "pro" else "pro"
        return next_speaker, next_team

    def _all_have_spoken(self, speak_counts: dict, agents: dict) -> bool:
        """Check if all required debaters have spoken at least once.

        Args:
            speak_counts: Count of speeches per agent
            agents: Available agents

        Returns:
            True if all available agents have spoken
        """
        for agent_id in self._REQUIRED_SPEAKERS:
            if agent_id in agents and speak_counts.get(agent_id, 0) == 0:
                return False
        return True

    def execute(
        self,
        pool,
        agents: dict,
        penalties: dict | None = None,
    ) -> dict:
        """Execute free debate stage.

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent
            penalties: Optional penalty configuration

        Returns:
            Result dictionary with status and turn count
        """
        self._display.stage_start(self.name, self.description)

        messages_published = 0
        speak_counts: dict[str, int] = {}
        turn_count = 0
        max_turns = 20  # Prevent infinite loops

        # Initialize timers for both teams
        pro_timer = Timer(
            total_seconds=self._TEAM_TIME,
            chars_per_minute=self._CHARS_PER_MINUTE,
        )
        con_timer = Timer(
            total_seconds=self._TEAM_TIME,
            chars_per_minute=self._CHARS_PER_MINUTE,
        )

        # Pro starts
        current_team = "pro"
        last_speaker = None

        while turn_count < max_turns:
            # Check if both teams out of time
            if pro_timer.is_expired() and con_timer.is_expired():
                logger.info("Both teams out of time, ending free debate")
                break

            # Check if current team has time
            current_timer = pro_timer if current_team == "pro" else con_timer
            if current_timer.is_expired():
                current_team = "con" if current_team == "pro" else "pro"
                current_timer = pro_timer if current_team == "pro" else con_timer
                if current_timer.is_expired():
                    break

            # Select next speaker
            next_speaker, next_team = self._get_next_speaker(
                agents, current_team, speak_counts, last_speaker
            )

            if next_speaker is None:
                break

            speaker = agents[next_speaker]

            # Generate speech with streaming
            recent_context = self._get_recent_context(pool, limit=5)
            self._display.speech(
                speaker=speaker.name,
                content="正在思考...",
                word_count=0,
                expected=100,  # Approximate
            )

            # Callback for streaming output
            content_buffer = []
            def stream_callback(char: str) -> None:
                content_buffer.append(char)

            content = speaker.generate_free_debate_speech(
                pool,
                recent_context=recent_context,
                callback=stream_callback,
            )
            word_count = len(content)

            # Check time
            is_overtime, elapsed = current_timer.check(word_count)
            time_left = current_timer.time_left()

            # Publish message
            message = Message(
                speaker=speaker.agent_id,
                role=speaker.role,
                team=speaker.team,
                stage=self.name,
                content=content,
                msg_type="free_speech",
                timestamp=time.time(),
                word_count=word_count,
                metadata=("turn", turn_count, "overtime", is_overtime),
            )
            pool.publish("public", message)
            messages_published += 1

            # Display speech with time info
            chars_left = current_timer.char_limit(time_left)
            time_used_total = (self._TEAM_TIME - time_left)

            # Use streaming display if content was generated with streaming
            if content_buffer:
                self._display.speech_stream(
                    speaker=speaker.name,
                    content=content,
                    word_count=word_count,
                    expected=chars_left + word_count,
                    time_used=time_used_total,
                    time_limit=self._TEAM_TIME,
                )
            else:
                self._display.speech(
                    speaker=speaker.name,
                    content=content,
                    word_count=word_count,
                    expected=chars_left + word_count,
                    time_used=time_used_total,
                    time_limit=self._TEAM_TIME,
                )

            # Update tracking
            speak_counts[next_speaker] = speak_counts.get(next_speaker, 0) + 1
            last_speaker = next_speaker
            current_team = next_team
            turn_count += 1

            # Check if everyone has spoken (minimum requirement met)
            if turn_count >= 8 and self._all_have_spoken(speak_counts, agents):
                # Allow a few more rounds for back-and-forth
                if turn_count >= 12:
                    break

        self._display.stage_end(self.name)

        return {
            "status": "completed",
            "stage": self.name,
            "messages_count": messages_published,
            "turns": turn_count,
            "speak_counts": speak_counts,
            "pro_time_left": pro_timer.time_left(),
            "con_time_left": con_timer.time_left(),
        }

    def execute_concurrent(
        self,
        pool,
        agents: dict,
        penalties: dict | None = None,
    ) -> dict:
        """Execute free debate with concurrent per-round parallel generation.

        Each round, one pro speaker and one con speaker generate simultaneously
        in separate threads. Both speeches are committed to the pool after the
        round completes (both threads reach the Barrier).

        Args:
            pool: MessagePool instance
            agents: Dictionary of agent_id -> Agent
            penalties: Optional penalty configuration

        Returns:
            Result dictionary with status, turn count, and concurrent=True
        """
        self._display.stage_start(self.name, self.description + " [并发模式]")

        messages_published = 0
        speak_counts: dict[str, int] = {}
        round_count = 0
        max_rounds = 10

        # Per-team time tracking
        pro_timer = Timer(total_seconds=self._TEAM_TIME, chars_per_minute=self._CHARS_PER_MINUTE)
        con_timer = Timer(total_seconds=self._TEAM_TIME, chars_per_minute=self._CHARS_PER_MINUTE)

        # Per-team last-speaker tracking (separate, not shared)
        pro_last: str | None = None
        con_last: str | None = None

        while round_count < max_rounds:
            # Check termination
            if pro_timer.is_expired() and con_timer.is_expired():
                break

            # Select speakers for this round
            pro_speaker_id, _ = self._get_next_speaker(agents, "pro", speak_counts, pro_last)
            con_speaker_id, _ = self._get_next_speaker(agents, "con", speak_counts, con_last)

            if pro_speaker_id is None and con_speaker_id is None:
                break

            # Results accumulated from threads
            pro_content: list[str] = [""]   # mutable container
            con_content: list[str] = [""]
            pro_error: list[Exception | None] = [None]
            con_error: list[Exception | None] = [None]

            # Streaming buffers (one per speaker)
            pro_buf = io.StringIO()
            con_buf = io.StringIO()
            buf_lock = threading.Lock()

            barrier = threading.Barrier(2)

            def _run_speaker(
                speaker_id: str | None,
                timer,
                buf: io.StringIO,
                content_out: list[str],
                error_slot: list,
                _barrier: threading.Barrier,
                lock: threading.Lock,
            ) -> None:
                if speaker_id is None or timer.is_expired():
                    with lock:
                        buf.write("⏰ 时间到")
                    try:
                        _barrier.wait()
                    except threading.BrokenBarrierError:
                        pass
                    return
                try:
                    speaker = agents[speaker_id]
                    recent_context = self._get_recent_context(pool, limit=5)

                    def _callback(char: str) -> None:
                        with lock:
                            buf.write(char)

                    content = speaker.generate_free_debate_speech(
                        pool,
                        recent_context=recent_context,
                        callback=_callback,
                    )
                    content_out[0] = content
                    _barrier.wait()
                except threading.BrokenBarrierError:
                    pass  # sibling aborted
                except Exception as e:
                    error_slot[0] = e
                    logger.warning("Concurrent speaker %s failed: %s", speaker_id, e)
                    try:
                        _barrier.abort()
                    except Exception:
                        pass

            # Launch threads
            pro_thread = threading.Thread(
                target=_run_speaker,
                args=(pro_speaker_id, pro_timer, pro_buf, pro_content, pro_error, barrier, buf_lock),
                daemon=True,
            )
            con_thread = threading.Thread(
                target=_run_speaker,
                args=(con_speaker_id, con_timer, con_buf, con_content, con_error, barrier, buf_lock),
                daemon=True,
            )
            pro_thread.start()
            con_thread.start()

            # Display side-by-side while threads run
            self._display.concurrent_speech_panels(
                pro_name=agents[pro_speaker_id].name if pro_speaker_id else "正方",
                con_name=agents[con_speaker_id].name if con_speaker_id else "反方",
                pro_buf=pro_buf,
                con_buf=con_buf,
                pro_thread=pro_thread,
                con_thread=con_thread,
                buf_lock=buf_lock,
            )

            pro_thread.join()
            con_thread.join()

            # Commit results (both or neither — atomic round)
            pro_text = pro_content[0]
            con_text = con_content[0]

            committed = 0
            if pro_text and pro_speaker_id and not pro_timer.is_expired():
                pro_speaker = agents[pro_speaker_id]
                word_count = len(pro_text)
                pro_timer.check(word_count)
                msg = Message(
                    speaker=pro_speaker.agent_id,
                    role=pro_speaker.role,
                    team=pro_speaker.team,
                    stage=self.name,
                    content=pro_text,
                    msg_type="free_speech",
                    timestamp=time.time(),
                    word_count=word_count,
                    metadata=("round", round_count, "concurrent", True),
                )
                pool.publish("public", msg)
                speak_counts[pro_speaker_id] = speak_counts.get(pro_speaker_id, 0) + 1
                pro_last = pro_speaker_id
                committed += 1

            if con_text and con_speaker_id and not con_timer.is_expired():
                con_speaker = agents[con_speaker_id]
                word_count = len(con_text)
                con_timer.check(word_count)
                msg = Message(
                    speaker=con_speaker.agent_id,
                    role=con_speaker.role,
                    team=con_speaker.team,
                    stage=self.name,
                    content=con_text,
                    msg_type="free_speech",
                    timestamp=time.time(),
                    word_count=word_count,
                    metadata=("round", round_count, "concurrent", True),
                )
                pool.publish("public", msg)
                speak_counts[con_speaker_id] = speak_counts.get(con_speaker_id, 0) + 1
                con_last = con_speaker_id
                committed += 1

            messages_published += committed
            round_count += 1

            # Early stop: all have spoken and enough rounds done
            if round_count >= 6 and self._all_have_spoken(speak_counts, agents):
                break

        self._display.stage_end(self.name)

        return {
            "status": "completed",
            "stage": self.name,
            "messages_count": messages_published,
            "rounds": round_count,
            "speak_counts": speak_counts,
            "concurrent": True,
            "pro_time_left": pro_timer.time_left(),
            "con_time_left": con_timer.time_left(),
        }

    def _get_recent_context(self, pool, limit: int = 5) -> str:
        """Get recent debate context for next speaker.

        Args:
            pool: MessagePool instance
            limit: Number of recent messages

        Returns:
            Formatted recent context string
        """
        recent = pool.get_messages("public")[-limit:] if limit > 0 else []
        if not recent:
            return "暂无发言"

        lines = [f"{m.speaker}：{m.content[:50]}..." for m in recent]
        return "\n".join(lines)
