"""Turning rows into a design matrix, and deciding what the model may see."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from airquality.data import COLOCATED_SENSOR, TARGET, Row


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Which columns the model gets, and what it is being asked to do.

    `honest` excludes the co-located tin-oxide sensor. With it, the task is not
    "predict benzene from meteorology and other pollutants", it is "convert one
    calibrated reading of benzene into another", and an R-squared near 1.0 says
    nothing about air quality.
    """

    name: str
    columns: tuple[str, ...]
    honest: bool

    def __len__(self) -> int:
        return len(self.columns)


def build(
    available: Sequence[str],
    *,
    include_colocated: bool,
    include_hour: bool = True,
) -> FeatureSet:
    columns = tuple(
        c
        for c in available
        if c != TARGET and (include_colocated or c != COLOCATED_SENSOR)
    )
    if include_hour:
        columns += ("hour", "weekday")
    return FeatureSet(
        name="with co-located sensor"
        if include_colocated
        else "without co-located sensor",
        columns=columns,
        honest=not include_colocated,
    )


def matrix(rows: Sequence[Row], features: FeatureSet) -> list[list[float]]:
    """Design matrix. Rows must already be complete for these columns."""
    built = []
    for row in rows:
        values = []
        for column in features.columns:
            if column == "hour":
                values.append(float(row.timestamp.hour))
            elif column == "weekday":
                values.append(float(row.timestamp.weekday()))
            else:
                value = row.values.get(column)
                if value is None:
                    raise ValueError(
                        f"missing {column!r} at {row.timestamp}; filter or impute first"
                    )
                values.append(float(value))
        built.append(values)
    return built


def target(rows: Sequence[Row], name: str = TARGET) -> list[float]:
    values = []
    for row in rows:
        value = row.values.get(name)
        if value is None:
            raise ValueError(f"missing target at {row.timestamp}")
        values.append(float(value))
    return values


def standardise(
    train: Sequence[Sequence[float]], test: Sequence[Sequence[float]]
) -> tuple[list[list[float]], list[list[float]]]:
    """Scale using the *training* mean and spread only.

    Fitting the scaler on all the data before splitting is the third leak in
    this dataset's usual write-up, alongside the shuffled split and the
    co-located sensor. It is the smallest of the three and the easiest to miss,
    because nothing about the resulting numbers looks wrong.
    """
    if not train:
        raise ValueError("cannot standardise on an empty training set")
    width = len(train[0])
    means = [sum(row[j] for row in train) / len(train) for j in range(width)]
    spreads = []
    for j in range(width):
        variance = sum((row[j] - means[j]) ** 2 for row in train) / max(len(train) - 1, 1)
        # A constant column has no spread; dividing by it produces NaN for every
        # row. Leaving it at 1.0 keeps it constant and harmless.
        spreads.append(variance**0.5 or 1.0)

    def apply(data: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(row[j] - means[j]) / spreads[j] for j in range(width)] for row in data]

    return apply(train), apply(test)
