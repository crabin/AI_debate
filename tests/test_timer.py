from src.engine.timer import Timer


def test_estimate_duration_chinese():
    timer = Timer(chars_per_minute=250)
    text = "你" * 250
    assert timer.estimate_duration(text) == 60.0


def test_check_overtime_within_limit():
    timer = Timer(chars_per_minute=250)
    text = "你" * 200  # 200 chars = 48 seconds
    is_over, excess = timer.check_overtime(text, limit_seconds=60)
    assert is_over is False
    assert excess == 0.0


def test_check_overtime_exceeded():
    timer = Timer(chars_per_minute=250)
    text = "你" * 300  # 300 chars = 72 seconds
    is_over, excess = timer.check_overtime(text, limit_seconds=60)
    assert is_over is True
    assert abs(excess - 12.0) < 0.1


def test_char_limit_for_duration():
    timer = Timer(chars_per_minute=250)
    assert timer.char_limit(seconds=180) == 750  # 3 minutes = 750 chars
    assert timer.char_limit(seconds=30) == 125   # 30 seconds = 125 chars
    assert timer.char_limit(seconds=120) == 500  # 2 minutes = 500 chars


def test_is_warning_zone():
    timer = Timer(chars_per_minute=250, warning_threshold=30)
    # 750 char limit, used 730 chars -> 20 chars remaining < 30 threshold
    assert timer.is_warning_zone(used_chars=730, limit_chars=750) is True
    # used 700 chars -> 50 remaining > 30 threshold
    assert timer.is_warning_zone(used_chars=700, limit_chars=750) is False
