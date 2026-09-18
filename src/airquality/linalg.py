"""Ordinary least squares in the standard library.

Written out rather than imported so the leakage demonstration runs with no
third-party dependency at all, and because the failure mode that matters here —
a singular normal-equations matrix when two columns are near-duplicates — is
exactly the situation the experiment creates on purpose.
"""

from __future__ import annotations

from collections.abc import Sequence


class SingularMatrix(ValueError):
    """The system has no unique solution."""


def solve(
    matrix: list[list[float]], rhs: list[float], *, tolerance: float = 1e-12
) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(matrix)
    if any(len(row) != n for row in matrix) or len(rhs) != n:
        raise ValueError("solve expects a square matrix and a matching right-hand side")

    augmented = [[*row, value] for row, value in zip(matrix, rhs, strict=True)]

    for column in range(n):
        # Partial pivoting: without it, a zero (or tiny) pivot silently produces
        # coefficients of 1e17 rather than an error.
        pivot_row = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot_row][column]) < tolerance:
            raise SingularMatrix(
                f"column {column} is linearly dependent on the others; "
                f"the design matrix has duplicate or collinear features"
            )
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        for row in range(column + 1, n):
            factor = augmented[row][column] / augmented[column][column]
            for k in range(column, n + 1):
                augmented[row][k] -= factor * augmented[column][k]

    solution = [0.0] * n
    for row in reversed(range(n)):
        total = augmented[row][n] - sum(
            augmented[row][k] * solution[k] for k in range(row + 1, n)
        )
        solution[row] = total / augmented[row][row]
    return solution


class LeastSquares:
    """y = Xb + intercept, fitted by the normal equations.

    Ridge regularisation is available and defaults to a small positive value.
    With zero it is textbook OLS, which is what the tests check against
    hand-computed values; in the experiment the features include a sensor that
    nearly duplicates the target, and an unregularised fit on near-collinear
    columns is what `SingularMatrix` exists to report.
    """

    def __init__(self, *, ridge: float = 0.0) -> None:
        if ridge < 0:
            raise ValueError(f"ridge must be non-negative; got {ridge}")
        self.ridge = ridge
        self.coefficients: list[float] = []
        self.intercept: float = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> LeastSquares:
        if len(x) != len(y):
            raise ValueError(f"x has {len(x)} rows and y has {len(y)} values")
        if not x:
            raise ValueError("cannot fit on an empty design matrix")

        n_features = len(x[0])
        if any(len(row) != n_features for row in x):
            raise ValueError("every row of the design matrix must have the same width")
        if len(x) <= n_features:
            # More parameters than observations: the fit is exact and means
            # nothing. Saying so beats returning a perfect training R-squared.
            raise ValueError(
                f"{len(x)} row(s) cannot determine {n_features} "
                "coefficient(s) plus an intercept"
            )

        # Centre so the intercept is not part of the normal equations; this
        # keeps the matrix better conditioned and the ridge term out of it.
        means = [sum(row[j] for row in x) / len(x) for j in range(n_features)]
        y_mean = sum(y) / len(y)
        centred = [[row[j] - means[j] for j in range(n_features)] for row in x]
        target = [value - y_mean for value in y]

        gram = [
            [
                sum(centred[i][a] * centred[i][b] for i in range(len(centred)))
                + (self.ridge if a == b else 0.0)
                for b in range(n_features)
            ]
            for a in range(n_features)
        ]
        moment = [
            sum(centred[i][a] * target[i] for i in range(len(centred)))
            for a in range(n_features)
        ]

        self.coefficients = solve(gram, moment)
        self.intercept = y_mean - sum(
            coefficient * mean
            for coefficient, mean in zip(self.coefficients, means, strict=True)
        )
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> list[float]:
        if not self.coefficients:
            raise RuntimeError("model is not fitted")
        if any(len(row) != len(self.coefficients) for row in x):
            raise ValueError("prediction rows must match the fitted feature count")
        return [
            self.intercept
            + sum(c * value for c, value in zip(self.coefficients, row, strict=True))
            for row in x
        ]


class NearestNeighbour:
    """k-nearest-neighbour regression, standard library, brute force.

    Here to make a point that a linear model cannot make. A shuffled split on
    an autocorrelated series leaks, but a linear model can barely exploit the
    leak — it has no way to memorise a neighbourhood. A model that *can* — a
    nearest-neighbour, a deep tree, a random forest — turns the same leak into
    a much larger number.

    So "how much does the shuffled split inflate the result" has no single
    answer: it depends on the capacity of the model being evaluated, which is
    why the experiment runs both.
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1; got {k}")
        self.k = k
        self._x: list[list[float]] = []
        self._y: list[float] = []

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> NearestNeighbour:
        if len(x) != len(y):
            raise ValueError(f"x has {len(x)} rows and y has {len(y)} values")
        if not x:
            raise ValueError("cannot fit on an empty training set")
        if len(x) < self.k:
            raise ValueError(
                f"k={self.k} needs at least {self.k} training rows, got {len(x)}"
            )
        self._x = [list(row) for row in x]
        self._y = list(y)
        return self

    def predict(self, x: Sequence[Sequence[float]]) -> list[float]:
        if not self._x:
            raise RuntimeError("model is not fitted")
        predictions = []
        for row in x:
            distances = sorted(
                (sum((a - b) ** 2 for a, b in zip(row, train, strict=True)), index)
                for index, train in enumerate(self._x)
            )
            nearest = distances[: self.k]
            predictions.append(sum(self._y[i] for _, i in nearest) / len(nearest))
        return predictions
