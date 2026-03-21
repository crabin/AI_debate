from src.engine.scorer import ScoreCard, Scorer


def test_scorecard_is_frozen():
    card = ScoreCard(
        speaker="pro_debater_1", stage="opening",
        logic=8, persuasion=7, expression=8, teamwork=7,
        rule_compliance=10, violations=(), comment="Good",
    )
    try:
        card.logic = 5
        assert False, "Should raise"
    except AttributeError:
        pass


def test_scorer_compute_speech_score():
    scorer = Scorer()
    card = ScoreCard(
        speaker="pro_debater_1", stage="opening",
        logic=8, persuasion=8, expression=8, teamwork=8,
        rule_compliance=8, violations=(), comment="Good",
    )
    score = scorer.compute_speech_score(card)
    assert score == 8.0  # All 8s with any weights = 8.0


def test_scorer_compute_weighted_score():
    scorer = Scorer()
    card = ScoreCard(
        speaker="pro_debater_1", stage="opening",
        logic=10, persuasion=10, expression=10, teamwork=10,
        rule_compliance=10, violations=(), comment="Perfect",
    )
    assert scorer.compute_speech_score(card) == 10.0


def test_scorer_record_and_get_individual_total():
    scorer = Scorer()
    card1 = ScoreCard("pro_debater_1", "opening", 8, 8, 8, 8, 8, (), "ok")
    card2 = ScoreCard("pro_debater_1", "cross_exam", 6, 6, 6, 6, 6, (), "ok")
    scorer.record(card1)
    scorer.record(card2)
    assert scorer.get_individual_total("pro_debater_1") == 14.0  # 8 + 6


def test_scorer_apply_individual_penalty():
    scorer = Scorer()
    card = ScoreCard("pro_debater_2", "cross_exam", 8, 8, 8, 8, 8, (), "ok")
    scorer.record(card)
    scorer.add_individual_penalty("pro_debater_2", -2, "overtime")
    assert scorer.get_individual_total("pro_debater_2") == 6.0  # 8 - 2


def test_scorer_apply_team_penalty():
    scorer = Scorer()
    scorer.add_team_penalty("pro", -3, "overtime")
    assert scorer.get_team_penalty("pro") == -3


def test_scorer_get_team_total():
    scorer = Scorer()
    card1 = ScoreCard("pro_debater_1", "opening", 8, 8, 8, 8, 8, (), "ok")
    card2 = ScoreCard("pro_debater_2", "cross_exam", 6, 6, 6, 6, 6, (), "ok")
    scorer.record(card1)
    scorer.record(card2)
    scorer.add_team_penalty("pro", -3, "overtime")
    # team total = (8 + 6) + (-3) = 11
    assert scorer.get_team_total("pro") == 11.0


def test_scorer_get_best_debater():
    scorer = Scorer()
    scorer.record(ScoreCard("pro_debater_1", "opening", 10, 10, 10, 10, 10, (), ""))
    scorer.record(ScoreCard("con_debater_1", "opening", 5, 5, 5, 5, 5, (), ""))
    best_id, best_score = scorer.get_best_debater()
    assert best_id == "pro_debater_1"
    assert best_score == 10.0


def test_scorer_get_stage_scores():
    scorer = Scorer()
    scorer.record(ScoreCard("pro_debater_1", "opening", 8, 8, 8, 8, 8, (), ""))
    scorer.record(ScoreCard("con_debater_1", "opening", 6, 6, 6, 6, 6, (), ""))
    stage_scores = scorer.get_stage_summary("opening")
    assert stage_scores["pro"] == 8.0
    assert stage_scores["con"] == 6.0
