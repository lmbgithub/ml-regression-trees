import math

import pytest

from airquality.metrics import mae, r2, rmse, score


def test_perfect_prediction():
    assert mae([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert rmse([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_mae_is_hand_checkable():
    assert mae([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]) == pytest.approx(1.0)


def test_rmse_punishes_one_large_error_more_than_mae():
    truth, predicted = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]
    assert mae(truth, predicted) == 1.0
    assert rmse(truth, predicted) == 2.0


def test_predicting_the_mean_scores_zero():
    truth = [1.0, 2.0, 3.0]
    assert r2(truth, [2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_r2_can_be_negative():
    # A model worse than the mean. Reporting it as 0 would hide that.
    assert r2([1.0, 2.0, 3.0], [10.0, 10.0, 10.0]) < 0


def test_r2_on_a_constant_target_is_undefined_not_zero():
    # "The model explains none of the variance" and "there was no variance"
    # are different statements, and a constant slice of a series produces the
    # second one.
    assert math.isnan(r2([5.0, 5.0, 5.0], [5.0, 4.0, 6.0]))


def test_r2_uses_the_mean_of_the_evaluated_set():
    # Not the training mean: scoring a test fold against the training mean
    # makes a shifted test set look far worse or better than it is.
    assert r2([10.0, 12.0], [10.0, 12.0]) == 1.0


@pytest.mark.parametrize("metric", [mae, rmse, r2])
def test_mismatched_lengths_are_refused(metric):
    with pytest.raises(ValueError, match="lengths differ"):
        metric([1.0, 2.0], [1.0])


@pytest.mark.parametrize("metric", [mae, rmse, r2])
def test_empty_input_is_refused(metric):
    with pytest.raises(ValueError, match="empty"):
        metric([], [])


def test_score_bundles_all_three_and_the_count():
    scores = score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert scores.n == 3 and scores.r2 == 1.0
    assert "MAE" in str(scores) and "R2" in str(scores)
