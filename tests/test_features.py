"""What the model is allowed to see."""

from datetime import datetime

import pytest

from airquality.data import COLOCATED_SENSOR, TARGET, Row
from airquality.features import build, matrix, standardise, target

COLUMNS = (TARGET, COLOCATED_SENSOR, "T", "RH")
ROW = Row(
    datetime(2004, 3, 10, 18),
    {TARGET: 11.9, COLOCATED_SENSOR: 1046.0, "T": 13.6, "RH": 48.9},
)


def test_the_target_is_never_a_feature():
    assert TARGET not in build(COLUMNS, include_colocated=True).columns


def test_the_honest_feature_set_excludes_the_colocated_sensor():
    features = build(COLUMNS, include_colocated=False)
    assert COLOCATED_SENSOR not in features.columns
    assert features.honest


def test_the_leaking_feature_set_includes_it_and_says_so():
    features = build(COLUMNS, include_colocated=True)
    assert COLOCATED_SENSOR in features.columns
    assert not features.honest


def test_calendar_features_are_derived_from_the_timestamp():
    features = build(COLUMNS, include_colocated=False, include_hour=True)
    row = matrix([ROW], features)[0]
    assert row[-2] == 18.0  # hour
    assert row[-1] == float(ROW.timestamp.weekday())


def test_calendar_features_can_be_turned_off():
    assert (
        "hour" not in build(COLUMNS, include_colocated=False, include_hour=False).columns
    )


def test_a_missing_value_is_an_error_not_a_zero():
    features = build(COLUMNS, include_colocated=False)
    incomplete = Row(ROW.timestamp, {**ROW.values, "T": None})
    with pytest.raises(ValueError, match="filter or impute"):
        matrix([incomplete], features)


def test_a_missing_target_is_an_error():
    with pytest.raises(ValueError, match="missing target"):
        target([Row(ROW.timestamp, {TARGET: None})])


def test_standardising_uses_the_training_statistics_only():
    # Fitting the scaler on all the data before splitting is the third leak,
    # and the easiest to miss because nothing about the numbers looks wrong.
    train = [[0.0], [2.0]]
    test = [[4.0]]
    scaled_train, scaled_test = standardise(train, test)
    assert scaled_train[0][0] == pytest.approx(-0.7071, abs=1e-3)
    assert scaled_test[0][0] > max(row[0] for row in scaled_train)


def test_a_constant_column_does_not_become_nan():
    scaled, _ = standardise([[5.0], [5.0], [5.0]], [[5.0]])
    assert all(row[0] == 0.0 for row in scaled)


def test_standardising_an_empty_training_set_is_refused():
    with pytest.raises(ValueError, match="empty training set"):
        standardise([], [[1.0]])
