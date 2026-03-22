import time
from src.engine.message_pool import Message, MessagePool


def test_publish_to_public():
    pool = MessagePool()
    msg = Message(
        speaker="pro_debater_1",
        role="一辩",
        team="pro",
        stage="opening",
        content="我方认为...",
        msg_type="speech",
        timestamp=time.time(),
        word_count=5,
        metadata=(),
    )
    pool.publish("public", msg)
    assert len(pool.get_messages("public")) == 1
    assert pool.get_messages("public")[0].content == "我方认为..."


def test_team_channel_isolation():
    pool = MessagePool()
    pro_msg = Message(
        speaker="pro_debater_1", role="一辩", team="pro", stage="opening",
        content="正方策略", msg_type="team_strategy",
        timestamp=time.time(), word_count=4, metadata=(),
    )
    con_msg = Message(
        speaker="con_debater_1", role="一辩", team="con", stage="opening",
        content="反方策略", msg_type="team_strategy",
        timestamp=time.time(), word_count=4, metadata=(),
    )
    pool.publish("team_pro", pro_msg)
    pool.publish("team_con", con_msg)
    assert len(pool.get_messages("team_pro")) == 1
    assert len(pool.get_messages("team_con")) == 1
    assert pool.get_messages("team_pro")[0].content == "正方策略"


def test_get_visible_messages_for_pro():
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("pro_debater_1", "一辩", "pro", "opening", "公开", "speech", ts, 2, ()))
    pool.publish("team_pro", Message("pro_debater_1", "一辩", "pro", "opening", "队内", "team_strategy", ts, 2, ()))
    pool.publish("team_con", Message("con_debater_1", "一辩", "con", "opening", "对方队内", "team_strategy", ts, 4, ()))

    visible = pool.get_visible_messages("pro")
    contents = [m.content for m in visible]
    assert "公开" in contents
    assert "队内" in contents
    assert "对方队内" not in contents


def test_get_visible_messages_for_judge():
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("pro_debater_1", "一辩", "pro", "opening", "公开", "speech", ts, 2, ()))
    pool.publish("judge_notes", Message("judge", "裁判", "judge", "opening", "笔记", "score", ts, 2, ()))
    pool.publish("team_pro", Message("pro_debater_1", "一辩", "pro", "opening", "队内", "team_strategy", ts, 2, ()))

    visible = pool.get_visible_messages("judge")
    contents = [m.content for m in visible]
    assert "公开" in contents
    assert "笔记" in contents
    assert "队内" not in contents


def test_message_is_frozen():
    msg = Message("a", "b", "c", "d", "e", "f", 0.0, 0, ())
    try:
        msg.content = "new"
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass


def test_get_messages_by_stage():
    pool = MessagePool()
    ts = time.time()
    pool.publish("public", Message("a", "一辩", "pro", "opening", "陈词", "speech", ts, 2, ()))
    pool.publish("public", Message("b", "二辩", "pro", "cross_exam", "攻辩", "speech", ts, 2, ()))

    opening_msgs = pool.get_messages("public", stage="opening")
    assert len(opening_msgs) == 1
    assert opening_msgs[0].content == "陈词"
