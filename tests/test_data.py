"""Reading the dataset — where -200 is the most dangerous number in the file."""

from datetime import datetime

import pytest

from airquality.data import MISSING_CODE, TARGET, DataError, Dataset, Row, parse_csv

CSV = (
    "Date;Time;CO(GT);C6H6(GT);PT08.S2(NMHC);NMHC(GT)\n"
    "10/03/2004;18.00.00;2,6;11,9;1046;150\n"
    "10/03/2004;19.00.00;2;9,4;955;-200\n"
    "10/03/2004;20.00.00;-200;9,0;939;-200\n"
    ";;;;;\n"
)


def test_rows_and_columns_are_read():
    dataset = parse_csv(CSV)
    assert len(dataset) == 3
    assert TARGET in dataset.columns


def test_decimal_commas_are_parsed():
    assert parse_csv(CSV).rows[0].values["CO(GT)"] == pytest.approx(2.6)


def test_timestamps_are_parsed_from_the_dotted_time_format():
    assert parse_csv(CSV).rows[0].timestamp == datetime(2004, 3, 10, 18, 0)


def test_the_missing_code_becomes_none():
    # -200 is a plausible-looking number two orders of magnitude from the real
    # range. Left as a value it is the single most consequential mistake
    # available on this dataset.
    assert parse_csv(CSV).rows[2].values["CO(GT)"] is None


def test_no_minus_two_hundred_survives_parsing():
    dataset = parse_csv(CSV)
    assert all(
        value != MISSING_CODE
        for row in dataset.rows
        for value in row.values.values()
        if value is not None
    )


def test_trailing_empty_rows_are_dropped():
    assert len(parse_csv(CSV)) == 3


def test_missing_fraction_is_computed_per_column():
    dataset = parse_csv(CSV)
    assert dataset.missing_fraction("NMHC(GT)") == pytest.approx(2 / 3)
    assert dataset.missing_fraction(TARGET) == 0.0


def test_mostly_empty_columns_are_dropped():
    # Imputing a 90%-missing column manufactures a column of one repeated
    # estimate and gives a tree something to split on.
    dataset = parse_csv(CSV).drop_sparse_columns(threshold=0.5)
    assert "NMHC(GT)" not in dataset.columns
    assert TARGET in dataset.columns


def test_dropping_columns_keeps_every_row():
    assert len(parse_csv(CSV).drop_sparse_columns()) == 3


def test_complete_rows_filters_on_the_named_columns_only():
    dataset = parse_csv(CSV)
    assert len(dataset.complete_rows(["CO(GT)"])) == 2
    assert len(dataset.complete_rows([TARGET])) == 3


def test_an_unknown_column_names_what_exists():
    with pytest.raises(DataError, match="no column"):
        parse_csv(CSV).column("O3")


def test_a_file_without_a_date_column_is_refused():
    with pytest.raises(DataError, match="Date"):
        parse_csv("A;B\n1;2\n")


def test_a_file_with_no_usable_rows_is_refused():
    with pytest.raises(DataError, match="no usable rows"):
        parse_csv("Date;Time;CO(GT)\n;;\n")


def test_an_unparseable_timestamp_is_refused():
    with pytest.raises(DataError, match="unrecognised timestamp"):
        parse_csv("Date;Time;CO(GT)\nnot-a-date;18.00.00;1\n")


def test_sorting_by_time_is_applied_rather_than_assumed():
    rows = (
        Row(datetime(2004, 3, 10, 20), {TARGET: 1.0}),
        Row(datetime(2004, 3, 10, 18), {TARGET: 2.0}),
    )
    ordered = Dataset((TARGET,), rows).sorted_by_time()
    assert [r.timestamp.hour for r in ordered.rows] == [18, 20]


def test_non_numeric_values_become_missing():
    dataset = parse_csv("Date;Time;CO(GT)\n10/03/2004;18.00.00;n/a\n")
    assert dataset.rows[0].values["CO(GT)"] is None
