import pytest

from airquality.cli import build_parser, main


def test_the_synthetic_series_is_the_default():
    args = build_parser().parse_args([])
    assert not args.csv and not args.uci


def test_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--uci", "--synthetic"])


def test_a_run_on_the_synthetic_series(capsys):
    assert main(["--hours", "600", "--model", "ols"]) == 0
    out = capsys.readouterr().out
    assert "with co-located sensor" in out
    assert "honest arrangement" in out


def test_a_missing_csv_is_a_usage_error(tmp_path, capsys):
    assert main(["--csv", str(tmp_path / "absent.csv")]) == 2
    assert "file not found" in capsys.readouterr().err


def test_a_malformed_csv_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("A;B\n1;2\n")
    assert main(["--csv", str(path)]) == 2
    assert "Date" in capsys.readouterr().err


def test_a_real_csv_runs(tmp_path, capsys):
    path = tmp_path / "small.csv"
    rows = [
        f"10/03/2004;{hour:02d}.00.00;{2 + hour % 3},5;{10 + hour % 7},1;{900 + hour};{20 + hour}"
        for hour in range(24)
    ]
    path.write_text(
        "Date;Time;CO(GT);C6H6(GT);PT08.S2(NMHC);T\n" + "\n".join(rows) + "\n"
    )
    assert main(["--csv", str(path), "--model", "ols"]) == 0
    assert "source:" in capsys.readouterr().out


def test_repeating_needs_the_generator(tmp_path, capsys):
    path = tmp_path / "small.csv"
    path.write_text("Date;Time;C6H6(GT)\n10/03/2004;18.00.00;9,4\n")
    assert main(["--csv", str(path), "--runs", "3"]) == 2
    assert "only applies to the synthetic series" in capsys.readouterr().err


def test_repeating_reports_each_leak_with_its_spread(capsys):
    assert main(["--runs", "3", "--hours", "500", "--model", "ols"]) == 0
    out = capsys.readouterr().out
    assert "over 3 independent series" in out
    assert "co-located sensor" in out
