"""Run the leakage experiment, on the real dataset or the synthetic one."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from airquality import synthetic
from airquality.data import DataError, load_csv, load_uci
from airquality.experiment import effects_summary, repeat, run, table, verdict
from airquality.linalg import LeastSquares, NearestNeighbour

MODELS = {
    "ols": ("LeastSquares", lambda: LeastSquares()),
    "knn": ("NearestNeighbour", lambda: NearestNeighbour(k=1)),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airquality",
        description=(
            "What three kinds of leakage are worth, in R-squared, on the same data."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="path to AirQualityUCI.csv")
    source.add_argument(
        "--uci", action="store_true", help="download via ucimlrepo (id 360)"
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="use the generated series (default; needs no download)",
    )
    parser.add_argument(
        "--hours", type=int, default=4000, help="length of the synthetic series"
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=[*MODELS, "both"],
        default="both",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="repeat over N independent synthetic series and report each leak's "
        "effect with its spread (the only form in which the comparison means anything)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.runs and (args.csv or args.uci):
        print(
            "--runs needs a generator, so it only applies to the synthetic series.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.csv:
            dataset = load_csv(args.csv)
            source = str(args.csv)
        elif args.uci:
            dataset = load_uci()
            source = "UCI id 360"
        else:
            dataset = synthetic.generate(args.hours, seed=args.seed)
            source = f"synthetic ({args.hours} hours, seed {args.seed})"
    except FileNotFoundError:
        print(f"file not found: {args.csv}", file=sys.stderr)
        return 2
    except DataError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"source: {source}   rows: {len(dataset):,}   columns: {len(dataset.columns)}\n"
    )

    names = list(MODELS) if args.model == "both" else [args.model]
    arrangements = []
    for name in names:
        _, factory = MODELS[name]
        arrangements += run(
            dataset,
            model_factory=factory,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    print(table(arrangements))

    if args.runs:
        for name in names:
            label, factory = MODELS[name]
            print(f"\n=== {label}, repeated")
            print(
                effects_summary(
                    repeat(runs=args.runs, hours=args.hours, model_factory=factory),
                    args.runs,
                )
            )
        return 0
    for name in names:
        label, _ = MODELS[name]
        subset = [a for a in arrangements if a.model == label]
        if subset:
            print()
            print(verdict(subset))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
