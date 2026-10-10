from src.evaluate_promote import should_promote


def test_first_model_promoted_if_beats_baseline():
    assert should_promote(1.0, None, 2.0, 0.01)[0]


def test_never_promote_worse_than_baseline():
    assert not should_promote(2.5, None, 2.0, 0.01)[0]


def test_needs_min_improvement():
    assert not should_promote(0.995, 1.0, 2.0, 0.01)[0]
    assert should_promote(0.98, 1.0, 2.0, 0.01)[0]
