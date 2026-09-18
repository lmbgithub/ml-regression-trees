"""The split that produces the headline result."""

from datetime import datetime, timedelta

import pytest

from airquality.data import Dataset, Row
from airquality.split import chronological, gap, shuffled

START = datetime(2004, 3, 10, 18)


def series(n=100):
    rows = tuple(Row(START + timedelta(hours=i), {"y": float(i)}) for i in range(n))
    return Dataset(("y",), rows)


def test_chronological_sizes():
    split = chronological(series(100), test_fraction=0.25)
    assert split.sizes == (75, 25)


def test_chronological_puts_the_future_in_test():
    split = chronological(series(100))
    assert max(r.timestamp for r in split.train) < min(r.timestamp for r in split.test)


def test_chronological_does_not_leak_time():
    assert not chronological(series(100)).leaks_time


def test_a_shuffled_split_leaks_time_by_construction():
    assert shuffled(series(100)).leaks_time


def test_a_shuffled_split_still_uses_every_row_once():
    split = shuffled(series(60))
    assert len({r.timestamp for r in (*split.train, *split.test)}) == 60


def test_the_same_seed_gives_the_same_shuffled_split():
    assert [r.timestamp for r in shuffled(series(50), seed=3).test] == [
        r.timestamp for r in shuffled(series(50), seed=3).test
    ]


def test_different_seeds_give_different_shuffled_splits():
    assert [r.timestamp for r in shuffled(series(200), seed=1).test] != [
        r.timestamp for r in shuffled(series(200), seed=2).test
    ]


def test_chronological_sorts_before_splitting():
    # The published file happens to be ordered. A split that relies on that is
    # a split that breaks silently when the file changes.
    rows = tuple(
        Row(START + timedelta(hours=i), {"y": float(i)}) for i in reversed(range(40))
    )
    split = chronological(Dataset(("y",), rows))
    assert split.train[0].timestamp < split.train[-1].timestamp


def test_the_gap_between_train_and_test_is_reported():
    assert gap(chronological(series(100))) == pytest.approx(1.0)


def test_a_shuffled_split_has_no_meaningful_gap():
    assert gap(shuffled(series(100))) <= 0


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 2.0])
def test_out_of_range_fractions_are_refused(fraction):
    with pytest.raises(ValueError, match="strictly between"):
        chronological(series(50), test_fraction=fraction)


def test_a_fraction_that_rounds_to_an_empty_split_is_refused():
    with pytest.raises(ValueError, match="empty split"):
        chronological(series(3), test_fraction=0.01)
