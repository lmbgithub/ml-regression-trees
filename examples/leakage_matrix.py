"""What three kinds of leakage are worth, measured. No download, no dependencies.

The series is generated, so the honest ceiling is known — which is the only way
to tell a good model from a leaking one.

    python examples/leakage_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airquality.experiment import (
    effects_summary,
    repeat,
    run,
    table,
    verdict,
)
from airquality.linalg import NearestNeighbour
from airquality.synthetic import generate

# Nearest-neighbour prediction is O(train x test), so the repeated measurement
# uses a shorter series for it. The spread is what matters here, not the
# absolute ceiling, and it is stable at this length.
RUNS = 12
OLS_HOURS = 2000
KNN_HOURS = 800


def main() -> None:
    dataset = generate(2000, seed=0)
    arrangements = run(dataset, seed=0) + run(
        dataset, model_factory=lambda: NearestNeighbour(k=1), seed=0
    )
    print(table(arrangements))
    print()
    print(verdict([a for a in arrangements if a.model == "LeastSquares"]))

    print(f"\n{'=' * 80}")
    print("One run cannot rank the leaks. Repeating over independent series:\n")
    for label, factory, hours in (
        ("LeastSquares", None, OLS_HOURS),
        ("NearestNeighbour", lambda: NearestNeighbour(k=1), KNN_HOURS),
    ):
        effects = (
            repeat(runs=RUNS, hours=hours)
            if factory is None
            else repeat(runs=RUNS, hours=hours, model_factory=factory)
        )
        print(f"--- {label}  ({hours} hours x {RUNS} series)")
        print(effects_summary(effects, RUNS))
        print()


if __name__ == "__main__":
    main()
