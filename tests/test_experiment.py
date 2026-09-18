"""The 2x2, run on the synthetic series where the honest ceiling is known."""

import pytest

from airquality.experiment import run, table, verdict
from airquality.linalg import NearestNeighbour
from airquality.synthetic import generate


@pytest.fixture(scope="module")
def dataset():
    return generate(2000, seed=7)


@pytest.fixture(scope="module")
def arrangements(dataset):
    return run(dataset, seed=7)


def test_all_four_arrangements_run(arrangements):
    assert len(arrangements) == 4
    assert {(a.split_kind, a.honest) for a in arrangements} == {
        ("shuffled", True),
        ("shuffled", False),
        ("chronological", True),
        ("chronological", False),
    }


def test_only_one_arrangement_is_leak_free(arrangements):
    assert sum(1 for a in arrangements if not a.leaks) == 1


def test_the_colocated_sensor_inflates_r2_enormously(arrangements):
    honest = next(a for a in arrangements if not a.leaks)
    leaking = next(
        a for a in arrangements if a.split_kind == "chronological" and not a.honest
    )
    assert leaking.test_scores.r2 - honest.test_scores.r2 > 0.3


def test_the_honest_arrangement_is_far_from_perfect(arrangements):
    # The generator's signal-to-noise puts a real ceiling here. An honest
    # model that scored 0.98 would mean the generator was leaking too.
    honest = next(a for a in arrangements if not a.leaks)
    assert 0.2 < honest.test_scores.r2 < 0.85


def test_a_single_run_cannot_settle_the_shuffled_split_effect(arrangements):
    # On this seed the shuffled split makes the score *worse*. That is the
    # result, not a bug: the effect is smaller than its own run-to-run spread,
    # which is exactly why `repeat` exists.
    honest = next(a for a in arrangements if not a.leaks)
    shuffled_honest = next(
        a for a in arrangements if a.split_kind == "shuffled" and a.honest
    )
    gap = shuffled_honest.test_scores.r2 - honest.test_scores.r2
    assert abs(gap) < 0.3


def test_a_memorising_model_fits_its_training_set_perfectly(dataset):
    results = run(dataset, model_factory=lambda: NearestNeighbour(k=1), seed=7)
    assert all(a.train_scores.r2 == pytest.approx(1.0) for a in results)


def test_the_table_marks_the_leaking_rows(arrangements):
    text = table(arrangements)
    assert text.count("YES") == 3
    assert "no" in text


def test_the_verdict_quantifies_each_leak(arrangements):
    text = verdict(arrangements)
    assert "co-located sensor" in text
    assert "shuffled split" in text
    assert "R2" in text


def test_the_verdict_refuses_to_rank_the_leaks_without_measuring(arrangements):
    assert "takes a measurement" in verdict(arrangements)


def test_an_empty_run_does_not_crash():
    assert verdict([]) == "no arrangements ran"


def test_results_are_reproducible(dataset):
    first = run(dataset, seed=3)
    second = run(dataset, seed=3)
    assert [a.test_scores.r2 for a in first] == [a.test_scores.r2 for a in second]


# --- repeated measurement ---------------------------------------------------


@pytest.fixture(scope="module")
def effects():
    from airquality.experiment import repeat

    return repeat(runs=8, hours=1200)


def test_the_sensor_leak_is_consistent_across_series(effects):
    sensor = next(e for e in effects if e.name == "co-located sensor")
    assert sensor.mean > 0.3
    assert not sensor.crosses_zero
    assert min(sensor.deltas) > 0


def test_the_split_leak_is_smaller_than_its_own_spread(effects):
    # The finding: two items from the same "this invalidates your result" list
    # are not the same size of problem on this data.
    split = next(e for e in effects if e.name == "shuffled split")
    sensor = next(e for e in effects if e.name == "co-located sensor")
    assert abs(split.mean) < sensor.mean
    assert split.spread >= abs(split.mean) / 2


def test_crossing_zero_is_reported_in_words(effects):
    from airquality.experiment import effects_summary

    text = effects_summary(effects, 8)
    assert "sign not established" in text or "consistent" in text
    assert "range" in text


def test_a_single_run_cannot_produce_a_spread():
    from airquality.experiment import repeat

    with pytest.raises(ValueError, match="at least 2 runs"):
        repeat(runs=1)


def test_the_spread_of_one_value_is_undefined():
    import math

    from airquality.experiment import LeakEffect

    assert math.isnan(LeakEffect("x", (0.5,)).spread)


def test_an_effect_that_never_changes_sign_is_not_crossing_zero():
    from airquality.experiment import LeakEffect

    assert not LeakEffect("x", (0.1, 0.2, 0.3)).crosses_zero
    assert LeakEffect("y", (-0.1, 0.2)).crosses_zero
