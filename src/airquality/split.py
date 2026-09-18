"""Splitting a time series — the decision that produces the headline result.

`train_test_split(..., shuffle=True)` is the reflex, and on hourly sensor data
it is leakage. Consecutive hours are nearly identical: a random split puts
14:00 in training and 15:00 in test, so the model is asked to interpolate
between points it has already seen rather than to predict anything. The honest
split is chronological — train on the earlier months, test on the later ones,
which is the only arrangement that matches how the model would be used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from airquality.data import Dataset, Row


@dataclass(frozen=True, slots=True)
class Split:
    train: tuple[Row, ...]
    test: tuple[Row, ...]
    kind: str

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.test)

    @property
    def leaks_time(self) -> bool:
        """Does any test observation precede a training observation?

        Reported rather than assumed: this is what distinguishes the two splits
        and it is checkable from the data alone.
        """
        if not self.train or not self.test:
            return False
        return min(r.timestamp for r in self.test) < max(r.timestamp for r in self.train)


def chronological(dataset: Dataset, *, test_fraction: float = 0.25) -> Split:
    """Earlier observations train, later observations test."""
    rows = dataset.sorted_by_time().rows
    cut = _cut(len(rows), test_fraction)
    return Split(train=rows[:cut], test=rows[cut:], kind="chronological")


def shuffled(dataset: Dataset, *, test_fraction: float = 0.25, seed: int = 0) -> Split:
    """A uniformly random split. Included to be measured, not to be used."""
    rows = list(dataset.rows)
    random.Random(seed).shuffle(rows)
    cut = _cut(len(rows), test_fraction)
    return Split(train=tuple(rows[:cut]), test=tuple(rows[cut:]), kind="shuffled")


def _cut(total: int, test_fraction: float) -> int:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be strictly between 0 and 1; got {test_fraction}"
        )
    cut = total - int(total * test_fraction)
    if cut <= 0 or cut >= total:
        raise ValueError(
            f"test_fraction {test_fraction} leaves an empty split for {total} row(s)"
        )
    return cut


def gap(split: Split) -> float:
    """Hours between the end of training and the start of test.

    Zero for a shuffled split by construction. For a chronological split it is
    one sampling interval, and widening it is the standard defence against
    autocorrelation carrying information across the boundary.
    """
    if not split.train or not split.test:
        return float("nan")
    delta = min(r.timestamp for r in split.test) - max(r.timestamp for r in split.train)
    return delta.total_seconds() / 3600.0
