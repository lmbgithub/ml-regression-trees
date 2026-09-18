"""Loading and cleaning the UCI Air Quality series.

Two properties of this dataset decide the entire experiment, and neither is
visible from the column names:

* Missing values are coded as **-200**, not as blanks. Read naively, a column
  that is 90% missing looks like a column with a strong negative signal.
* The `PT08.S*` columns are the tin-oxide sensors, each **calibrated against**
  the co-located reference analyser. `PT08.S2(NMHC)` and the target `C6H6(GT)`
  are two readings of the same physical quantity, and a model given both is
  not predicting benzene — it is copying it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

MISSING_CODE = -200.0
TARGET = "C6H6(GT)"

#: The tin-oxide sensor co-located with the benzene reference analyser. Keeping
#: it as a feature is the leak; the experiment measures what it is worth.
COLOCATED_SENSOR = "PT08.S2(NMHC)"

#: Columns that are timestamps rather than measurements.
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"


class DataError(ValueError):
    """The file is not the Air Quality dataset this experiment expects."""


@dataclass(frozen=True, slots=True)
class Row:
    """One hourly observation."""

    timestamp: datetime
    values: dict[str, float | None]  # None where the value was -200

    def feature(self, name: str) -> float | None:
        return self.values.get(name)


@dataclass(frozen=True, slots=True)
class Dataset:
    columns: tuple[str, ...]
    rows: tuple[Row, ...]

    def __len__(self) -> int:
        return len(self.rows)

    def missing_fraction(self, column: str) -> float:
        if not self.rows:
            return float("nan")
        return sum(1 for r in self.rows if r.values.get(column) is None) / len(self.rows)

    def drop_sparse_columns(self, threshold: float = 0.5) -> Dataset:
        """Drop columns whose values are mostly missing.

        On this dataset that removes `NMHC(GT)`, which is 90% absent. Imputing
        a column that is 90% missing does not recover information; it
        manufactures a column of one repeated estimate and gives a tree
        something to split on.
        """
        keep = tuple(c for c in self.columns if self.missing_fraction(c) < threshold)
        return Dataset(
            columns=keep,
            rows=tuple(
                Row(r.timestamp, {c: r.values[c] for c in keep}) for r in self.rows
            ),
        )

    def complete_rows(self, columns: Sequence[str]) -> Dataset:
        """Keep only rows with every listed column present."""
        rows = tuple(
            r for r in self.rows if all(r.values.get(c) is not None for c in columns)
        )
        return Dataset(self.columns, rows)

    def column(self, name: str) -> list[float | None]:
        if name not in self.columns:
            raise DataError(f"no column {name!r}; have {', '.join(self.columns)}")
        return [r.values[name] for r in self.rows]

    def sorted_by_time(self) -> Dataset:
        """Chronological order — required before any time-aware split.

        The published CSV is already in order, which is exactly why this is
        applied explicitly: a split that assumes an ordering the file happens
        to have is a split that breaks silently when the file changes.
        """
        return Dataset(self.columns, tuple(sorted(self.rows, key=lambda r: r.timestamp)))


def parse_csv(text: str, *, delimiter: str = ";", decimal: str = ",") -> Dataset:
    """Read the published CSV, which is semicolon-separated with decimal commas.

    The file also carries two trailing empty columns and a block of empty rows
    at the end; both are dropped here rather than surviving as an all-missing
    feature.
    """
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    fieldnames = [f for f in (reader.fieldnames or []) if f and f.strip()]
    if DATE_COLUMN not in fieldnames or TIME_COLUMN not in fieldnames:
        raise DataError(
            f"expected {DATE_COLUMN!r} and {TIME_COLUMN!r} columns; found {fieldnames}"
        )

    measurement_columns = tuple(
        c for c in fieldnames if c not in (DATE_COLUMN, TIME_COLUMN)
    )
    rows: list[Row] = []
    for record in reader:
        raw_date = (record.get(DATE_COLUMN) or "").strip()
        raw_time = (record.get(TIME_COLUMN) or "").strip()
        if not raw_date or not raw_time:
            continue  # the trailing empty block
        timestamp = _parse_timestamp(raw_date, raw_time)
        values = {
            column: _parse_value(record.get(column), decimal)
            for column in measurement_columns
        }
        rows.append(Row(timestamp, values))

    if not rows:
        raise DataError("no usable rows in the file")
    return Dataset(measurement_columns, tuple(rows))


def load_csv(path: str | Path, **kwargs) -> Dataset:
    return parse_csv(Path(path).read_text(encoding="utf-8-sig"), **kwargs)


def load_uci(dataset_id: int = 360) -> Dataset:
    """Fetch through `ucimlrepo`, if it is installed.

    Optional on purpose: the package downloads on import and needs the network,
    and nothing in the measurement code should require either.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "ucimlrepo is not installed; run `pip install ucimlrepo` or pass a CSV path. "
            "See the README for the direct download."
        ) from exc

    frame = fetch_ucirepo(id=dataset_id).data.features
    return parse_csv(frame.to_csv(sep=";", index=False), decimal=".")


def _parse_timestamp(raw_date: str, raw_time: str) -> datetime:
    time_text = raw_time.replace(".", ":")
    for date_format in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(f"{raw_date} {time_text}", f"{date_format} %H:%M:%S")
        except ValueError:
            continue
    raise DataError(f"unrecognised timestamp: {raw_date!r} {raw_time!r}")


def _parse_value(raw: str | None, decimal: str) -> float | None:
    if raw is None:
        return None
    text = raw.strip().replace(decimal, ".") if decimal != "." else raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    # -200 is the dataset's missing marker. Leaving it as a number is the
    # single most consequential mistake available on this dataset: it is a
    # plausible-looking value two orders of magnitude from the real range.
    return None if value == MISSING_CODE else value


def iter_columns(dataset: Dataset) -> Iterator[tuple[str, list[float | None]]]:
    for column in dataset.columns:
        yield column, dataset.column(column)
