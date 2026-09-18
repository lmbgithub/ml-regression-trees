"""Regression metrics, written out so each one's failure case is visible."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scores:
    mae: float
    rmse: float
    r2: float
    n: int

    def __str__(self) -> str:
        return (
            f"MAE {self.mae:7.3f}  RMSE {self.rmse:7.3f}  R2 {self.r2:7.4f}  n={self.n}"
        )


def _check(truth: Sequence[float], predicted: Sequence[float]) -> None:
    if len(truth) != len(predicted):
        raise ValueError(f"lengths differ: {len(truth)} and {len(predicted)}")
    if not truth:
        raise ValueError("cannot score an empty set")


def mae(truth: Sequence[float], predicted: Sequence[float]) -> float:
    _check(truth, predicted)
    return sum(abs(t - p) for t, p in zip(truth, predicted, strict=True)) / len(truth)


def rmse(truth: Sequence[float], predicted: Sequence[float]) -> float:
    _check(truth, predicted)
    return math.sqrt(
        sum((t - p) ** 2 for t, p in zip(truth, predicted, strict=True)) / len(truth)
    )


def r2(truth: Sequence[float], predicted: Sequence[float]) -> float:
    """1 - SS_res/SS_tot, against the mean of the *evaluated* set.

    Two properties worth keeping in mind, both of which the experiment relies
    on. R-squared can be negative — a model worse than predicting the mean —
    and it is undefined when the evaluated set has no variance, which is what a
    constant slice of a time series produces. NaN is returned there rather than
    a value, because "the model explains none of the variance" and "there was
    no variance" are different statements.
    """
    _check(truth, predicted)
    mean = sum(truth) / len(truth)
    ss_total = sum((t - mean) ** 2 for t in truth)
    if ss_total == 0:
        return float("nan")
    ss_residual = sum((t - p) ** 2 for t, p in zip(truth, predicted, strict=True))
    return 1 - ss_residual / ss_total


def score(truth: Sequence[float], predicted: Sequence[float]) -> Scores:
    return Scores(
        mae=mae(truth, predicted),
        rmse=rmse(truth, predicted),
        r2=r2(truth, predicted),
        n=len(truth),
    )
