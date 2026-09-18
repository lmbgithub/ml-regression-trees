"""A synthetic series with the same three leaks, built on purpose.

The real dataset is a download away, and the point of the experiment is a
comparison of *arrangements*, not a claim about Italian air quality. This
generator produces an hourly series where:

* the target is smooth and strongly autocorrelated, so a shuffled split can
  interpolate it almost perfectly;
* one feature is a co-located sensor — a noisy affine function of the target,
  exactly what a tin-oxide sensor calibrated against a reference analyser is;
* the remaining features carry a real but much weaker relationship, which is
  the honest signal.

Because the generating process is known, the honest ceiling is known too, which
is the only way to tell a leak from a good model.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from airquality.data import COLOCATED_SENSOR, TARGET, Dataset, Row

START = datetime(2004, 3, 10, 18, 0, 0)

#: Standard deviation of the co-located sensor's measurement noise, as a
#: fraction of the target's own spread. Small, because the real sensor is good —
#: which is precisely why using it is cheating.
SENSOR_NOISE = 0.05


def generate(
    hours: int = 4000,
    *,
    seed: int = 20260916,
    sensor_noise: float = SENSOR_NOISE,
    missing_fraction: float = 0.05,
) -> Dataset:
    rng = random.Random(seed)
    rows: list[Row] = []
    level = 10.0

    for index in range(hours):
        timestamp = START + timedelta(hours=index)

        # Daily cycle plus a slow random walk: the autocorrelation that makes a
        # shuffled split able to interpolate.
        daily = 4.0 * math.sin(2 * math.pi * timestamp.hour / 24)
        level = 0.97 * level + 0.03 * 10.0 + rng.gauss(0, 0.4)
        temperature = (
            15.0 + 8.0 * math.sin(2 * math.pi * (index / (24 * 90))) + rng.gauss(0, 1.5)
        )
        humidity = 50.0 - 0.8 * (temperature - 15.0) + rng.gauss(0, 6.0)

        # The honest signal: weak, real, and the most any model without the
        # co-located sensor can recover.
        benzene = max(
            0.1, level + daily - 0.15 * (temperature - 15.0) + rng.gauss(0, 1.2)
        )

        row_values: dict[str, float | None] = {
            TARGET: benzene,
            # Affine in the target: this is the leak.
            COLOCATED_SENSOR: 900.0 + 45.0 * benzene * (1 + rng.gauss(0, sensor_noise)),
            "T": temperature,
            "RH": humidity,
            "AH": 0.5 + 0.02 * temperature + rng.gauss(0, 0.05),
            "NOx(GT)": max(1.0, 120.0 + 20.0 * daily + rng.gauss(0, 40)),
        }

        for column in ("T", "RH", "NOx(GT)"):
            if rng.random() < missing_fraction:
                row_values[column] = None

        rows.append(Row(timestamp, row_values))

    return Dataset(columns=tuple(rows[0].values), rows=tuple(rows))
