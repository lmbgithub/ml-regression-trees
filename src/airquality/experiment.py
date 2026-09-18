"""The 2x2: shuffled or chronological split, with or without the co-located sensor."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from airquality import features
from airquality import split as split_module
from airquality.data import TARGET, Dataset
from airquality.linalg import LeastSquares, SingularMatrix
from airquality.metrics import Scores, score


@dataclass(frozen=True, slots=True)
class Arrangement:
    """One way of setting up the same problem."""

    model: str
    split_kind: str
    feature_set: str
    honest: bool
    train_scores: Scores
    test_scores: Scores
    leaks_time: bool
    n_features: int

    @property
    def leaks(self) -> bool:
        return self.leaks_time or not self.honest

    @property
    def label(self) -> str:
        return f"{self.split_kind} / {self.feature_set}"


def run(
    dataset: Dataset,
    *,
    model_factory: Callable[[], object] = LeastSquares,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> list[Arrangement]:
    """Fit the same model under all four arrangements."""
    cleaned = dataset.drop_sparse_columns()
    arrangements: list[Arrangement] = []

    for include_colocated in (True, False):
        feature_set = features.build(cleaned.columns, include_colocated=include_colocated)
        needed = [c for c in feature_set.columns if c not in ("hour", "weekday")] + [
            TARGET
        ]
        complete = cleaned.complete_rows(needed)

        for splitter, kind in (
            (
                lambda d: split_module.shuffled(
                    d, test_fraction=test_fraction, seed=seed
                ),
                "shuffled",
            ),
            (
                lambda d: split_module.chronological(d, test_fraction=test_fraction),
                "chronological",
            ),
        ):
            data_split = splitter(complete)
            x_train = features.matrix(data_split.train, feature_set)
            x_test = features.matrix(data_split.test, feature_set)
            y_train = features.target(data_split.train)
            y_test = features.target(data_split.test)
            x_train, x_test = features.standardise(x_train, x_test)

            model = model_factory()
            try:
                model.fit(x_train, y_train)
            except SingularMatrix:
                # Near-duplicate columns. Worth surfacing rather than silently
                # regularising away, since it is a symptom of the leak.
                continue

            arrangements.append(
                Arrangement(
                    model=getattr(model, "name", type(model).__name__),
                    split_kind=kind,
                    feature_set=feature_set.name,
                    honest=feature_set.honest,
                    train_scores=score(y_train, model.predict(x_train)),
                    test_scores=score(y_test, model.predict(x_test)),
                    leaks_time=data_split.leaks_time,
                    n_features=len(feature_set),
                )
            )

    return arrangements


def table(arrangements: Sequence[Arrangement]) -> str:
    header = (
        f"{'model':<18}{'split':<15}{'features':<28}{'test R2':>9}{'test RMSE':>11}"
        f"{'train R2':>10}{'leaks?':>8}"
    )
    lines = [header, "-" * len(header)]
    for a in sorted(arrangements, key=lambda a: (a.model, -a.test_scores.r2)):
        lines.append(
            f"{a.model:<18}{a.split_kind:<15}{a.feature_set:<28}{a.test_scores.r2:>9.4f}"
            f"{a.test_scores.rmse:>11.3f}{a.train_scores.r2:>10.4f}"
            f"{('YES' if a.leaks else 'no'):>8}"
        )
    return "\n".join(lines)


def verdict(arrangements: Sequence[Arrangement]) -> str:
    """State what each leak was worth, in R-squared."""
    if not arrangements:
        return "no arrangements ran"

    model_name = arrangements[0].model
    by_key = {(a.split_kind, a.honest): a for a in arrangements if a.model == model_name}
    honest = by_key.get(("chronological", True))
    if honest is None:
        return "the honest arrangement did not run"

    lines = [
        f"{model_name}: honest arrangement (chronological, no co-located sensor): "
        f"test R2 {honest.test_scores.r2:.4f}"
    ]

    sensor = by_key.get(("chronological", False))
    if sensor is not None:
        lines.append(
            f"  + co-located sensor      R2 {sensor.test_scores.r2:.4f} "
            f"({sensor.test_scores.r2 - honest.test_scores.r2:+.4f}) — "
            f"the sensor is a calibrated reading of the target"
        )

    shuffled_honest = by_key.get(("shuffled", True))
    if shuffled_honest is not None:
        lines.append(
            f"  + shuffled split         R2 {shuffled_honest.test_scores.r2:.4f} "
            f"({shuffled_honest.test_scores.r2 - honest.test_scores.r2:+.4f}) — "
            f"adjacent hours are nearly identical"
        )

    both = by_key.get(("shuffled", False))
    if both is not None:
        lines.append(
            f"  + both                   R2 {both.test_scores.r2:.4f} "
            f"({both.test_scores.r2 - honest.test_scores.r2:+.4f})"
        )

    lines.append(
        "\nEvery row above fits the same model on the same data. The only thing that "
        "changes is\nwhat the model is allowed to see and how the test set was chosen."
    )

    if sensor is not None and shuffled_honest is not None:
        sensor_cost = sensor.test_scores.r2 - honest.test_scores.r2
        split_cost = shuffled_honest.test_scores.r2 - honest.test_scores.r2
        ratio = sensor_cost / split_cost if abs(split_cost) > 1e-9 else float("inf")
        lines.append(
            "\nThe two leaks are not the same size: the co-located sensor is "
            f"worth {ratio:.1f}x what the\nshuffled split is worth for this "
            "model. Standard advice treats both as equally fatal.\nRanking them "
            "takes a measurement, and the ranking is a property of the data and "
            "the\nmodel's capacity to memorise, not of the advice."
        )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LeakEffect:
    """What one leak is worth, across repeated draws of the data.

    A single run cannot answer this. The effect of a leak is itself a random
    variable — it depends on which rows landed in the test set — and reporting
    one number for it is the same mistake the leak itself represents: treating
    an arrangement-specific result as a property of the problem.
    """

    name: str
    deltas: tuple[float, ...]

    @property
    def mean(self) -> float:
        return sum(self.deltas) / len(self.deltas) if self.deltas else float("nan")

    @property
    def spread(self) -> float:
        """Sample standard deviation; NaN with fewer than two runs."""
        import math

        if len(self.deltas) < 2:
            return float("nan")
        mean = self.mean
        return math.sqrt(
            sum((d - mean) ** 2 for d in self.deltas) / (len(self.deltas) - 1)
        )

    @property
    def crosses_zero(self) -> bool:
        """Did the leak help in some runs and hurt in others?

        When it does, a single run cannot establish even the *sign* of the
        effect, let alone its size — and any write-up quoting one number for it
        is quoting noise.
        """
        return min(self.deltas) < 0 < max(self.deltas) if self.deltas else False

    def __str__(self) -> str:
        verdict = "sign not established" if self.crosses_zero else "consistent"
        return (
            f"{self.name:<28}{self.mean:+.3f} +- {self.spread:.3f}   "
            f"range [{min(self.deltas):+.3f}, {max(self.deltas):+.3f}]   {verdict}"
        )


def repeat(
    *,
    runs: int = 12,
    hours: int = 2000,
    model_factory: Callable[[], object] = LeastSquares,
    generator: Callable[[int, int], Dataset] | None = None,
) -> list[LeakEffect]:
    """Measure each leak's effect over `runs` independent draws.

    The generator is reseeded per run, so each run is a fresh series rather than
    a fresh split of the same series: the question is what the leak is worth in
    general, not what it was worth on one sample.
    """
    if runs < 2:
        raise ValueError(f"a spread needs at least 2 runs; got {runs}")

    if generator is None:
        from airquality.synthetic import generate as default_generator

        def generator(hours: int, seed: int) -> Dataset:  # type: ignore[misc]
            return default_generator(hours, seed=seed)

    sensor_deltas: list[float] = []
    split_deltas: list[float] = []

    for seed in range(runs):
        arrangements = run(generator(hours, seed), model_factory=model_factory, seed=seed)
        by_key = {(a.split_kind, a.honest): a for a in arrangements}
        honest = by_key.get(("chronological", True))
        if honest is None:
            continue
        sensor = by_key.get(("chronological", False))
        shuffled_honest = by_key.get(("shuffled", True))
        if sensor is not None:
            sensor_deltas.append(sensor.test_scores.r2 - honest.test_scores.r2)
        if shuffled_honest is not None:
            split_deltas.append(shuffled_honest.test_scores.r2 - honest.test_scores.r2)

    return [
        LeakEffect("co-located sensor", tuple(sensor_deltas)),
        LeakEffect("shuffled split", tuple(split_deltas)),
    ]


def effects_summary(effects: Sequence[LeakEffect], runs: int) -> str:
    lines = [f"effect on test R2, over {runs} independent series:"]
    lines += [f"  {effect}" for effect in effects]

    unstable = [e for e in effects if e.crosses_zero]
    stable = [e for e in effects if not e.crosses_zero and e.deltas]
    if unstable and stable:
        lines.append(
            f"\n{stable[0].name} is worth {stable[0].mean:+.2f} R2 in every "
            f"run. {unstable[0].name} is smaller\nthan its own run-to-run "
            "spread — a single experiment cannot establish its sign, never\nmind "
            "its size. Both appear on the same list of things that invalidate a "
            "result; they\nare not the same size of problem on this data."
        )
    return "\n".join(lines)
