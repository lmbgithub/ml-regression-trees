# ml-regression-trees — three ways to get R² ≈ 0.98 on air quality

Linear regression and decision trees on UCI Air Quality. The usual write-up of
this dataset reports an almost-perfect fit and concludes that the model works.
It does not. There are three separate leaks available here, and this repository
measures what each one is worth.

The finding: **they are not the same size of problem.** One is worth +0.48 R² in
every single run. The other is smaller than its own run-to-run spread — a single
experiment cannot establish even its sign. Both appear on the same list of
things that invalidate a result.

**Standard library only. 101 tests.**

## What the run looks like

```
$ python examples/leakage_matrix.py
model             split          features                      test R2  test RMSE  train R2  leaks?
---------------------------------------------------------------------------------------------------
LeastSquares      shuffled       with co-located sensor         0.9789      0.546    0.9787     YES
LeastSquares      chronological  with co-located sensor         0.9683      0.593    0.9803     YES
LeastSquares      shuffled       without co-located sensor      0.6410      2.250    0.6093     YES
LeastSquares      chronological  without co-located sensor      0.5115      2.328    0.6193      no
NearestNeighbour  shuffled       with co-located sensor         0.8839      1.279    1.0000     YES
NearestNeighbour  chronological  with co-located sensor         0.8068      1.464    1.0000     YES
NearestNeighbour  shuffled       without co-located sensor      0.4372      2.817    1.0000     YES
NearestNeighbour  chronological  without co-located sensor      0.2844      2.817    1.0000      no

LeastSquares: honest arrangement (chronological, no co-located sensor): test R2 0.5115
  + co-located sensor      R2 0.9683 (+0.4567) — the sensor is a calibrated reading of the target
  + shuffled split         R2 0.6410 (+0.1295) — adjacent hours are nearly identical
  + both                   R2 0.9789 (+0.4673)

================================================================================
One run cannot rank the leaks. Repeating over independent series:

--- LeastSquares  (2000 hours x 12 series)
  co-located sensor           +0.484 +- 0.089   range [+0.379, +0.687]   consistent
  shuffled split              +0.084 +- 0.079   range [-0.011, +0.262]   sign not established

--- NearestNeighbour  (800 hours x 12 series)
  co-located sensor           +0.621 +- 0.162   range [+0.381, +0.998]   consistent
  shuffled split              +0.135 +- 0.153   range [-0.160, +0.440]   sign not established
```

Every row fits the same model on the same data. The only things that change are
what the model may see and how the test set was chosen.

## The seven decisions worth discussing

**1. `PT08.S2(NMHC)` is not a feature, it is the answer.** The `PT08.S*` columns
are tin-oxide sensors, each **calibrated against the co-located reference
analyser**. `PT08.S2(NMHC)` and the target `C6H6(GT)` are two readings of the
same physical quantity. A model given both is not predicting benzene from
meteorology and other pollutants; it is converting one calibrated reading into
another, and it will report R² ≈ 0.98 for doing so. Nothing in the column names
says this — it is in the dataset's documentation, and it is the whole result.

**2. A shuffled split on hourly data is leakage — worth less than you think.**
Consecutive hours are nearly identical, so a random split asks the model to
interpolate between points it has already seen. The received advice treats this
as fatal. Measured over twelve independent series it is worth +0.08 ± 0.08 for
OLS and +0.13 ± 0.15 for a memorising model: real on average, and **not
separable from noise in any one run**. Reporting a single number for it, as the
first version of this experiment did, is quoting the same kind of
arrangement-specific artifact that the leak itself is.

**3. Which is why one run is not the experiment.** `repeat()` regenerates the
series per run, so each measurement is a fresh sample rather than a fresh split
of the same sample, and reports mean, spread and range. `LeakEffect.crosses_zero`
says out loud when an effect changed sign between runs — the condition under
which no single-run claim about it is defensible.

**4. −200 is the missing-value code, not a measurement.** Read naively, a column
that is 90% missing looks like a column with a strong negative signal, two
orders of magnitude outside the real range. Parsing converts it to `None`, and a
test asserts no `-200` survives. Columns more than half missing (`NMHC(GT)`, 90%
absent) are dropped rather than imputed: imputing them manufactures a column of
one repeated estimate and gives a tree something to split on.

**5. The scaler is fitted on the training rows only.** The third leak, and the
easiest to miss because nothing about the resulting numbers looks wrong.
`features.standardise` takes train and test together precisely so it cannot be
called any other way by accident.

**6. The size of a leak depends on the model's capacity to memorise.** A
nearest-neighbour model fits its training set at R² = 1.0 by construction and
exposes a larger split leak than OLS. So "always use a time-based split" is
sound advice with no fixed price attached; the price is a property of the data
*and* the model, and is worth measuring.

**7. R² on a constant target is NaN, not zero.** "The model explains none of the
variance" and "there was no variance" are different statements, and a constant
slice of a time series produces the second one.

## Why the regression is written out

`LeastSquares` is about eighty lines of normal equations with partial pivoting,
and `NearestNeighbour` about thirty. That keeps the whole experiment — data,
split, features, model, metrics, repeated measurement — free of any third-party
dependency, so `python examples/leakage_matrix.py` runs on a clean interpreter
with no download.

It also makes one failure explicit that a library would smooth over: fitting OLS
on two near-duplicate columns raises `SingularMatrix` naming the collinearity,
rather than returning coefficients of 1e17. Given that this experiment creates
near-duplicate columns *on purpose*, that error is informative.

## Design

```
src/airquality/
  data.py        CSV parsing, the -200 code, sparse-column dropping
  split.py       chronological vs shuffled, and a leak check on the split itself
  features.py    the two feature sets, train-only standardisation
  linalg.py      OLS by normal equations + a k-NN, both standard library
  metrics.py     MAE, RMSE, R-squared with its undefined case
  synthetic.py   a series with the same leaks, so the honest ceiling is known
  experiment.py  the 2x2, and the repeated measurement of each leak
  cli.py         argument parsing
```

The synthetic generator is load-bearing, not a convenience. On the real dataset
you can show that removing the sensor lowers R², but you cannot show what the
*correct* number is. On a generated series the signal-to-noise is set
deliberately, so an honest model scoring 0.98 would mean the generator was
leaking too — and that is a test.

## Usage

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

```bash
pytest -q
python examples/leakage_matrix.py                 # no download
python -m airquality --runs 12                    # the repeated measurement
python -m airquality --csv AirQualityUCI.csv      # the real dataset
python -m airquality --uci                        # download via ucimlrepo
```

As a library:

```python
from airquality import load_csv, run, table
from airquality.experiment import effects_summary, repeat

print(table(run(load_csv("AirQualityUCI.csv"))))
print(effects_summary(repeat(runs=12), 12))
```

## Dataset

**UCI Air Quality** (id 360) — 9,357 hourly observations from a gas
multisensor device in an Italian city, March 2004 to February 2005. Not
included; about 2 MB.

Either download the CSV:

```bash
curl -O https://archive.ics.uci.edu/static/public/360/air+quality.zip
unzip air+quality.zip          # produces AirQualityUCI.csv
python -m airquality --csv AirQualityUCI.csv
```

or let `ucimlrepo` fetch it:

```bash
pip install -r requirements.txt
python -m airquality --uci
```

The published CSV is semicolon-separated with decimal commas, carries two
trailing empty columns and a block of empty rows, and codes missing values as
`-200`. All four are handled in `data.py`, and each has a test.

**The target is `C6H6(GT)`** — true hourly averaged benzene concentration from
the reference analyser. **The leak is `PT08.S2(NMHC)`** — the tin-oxide sensor
calibrated against it.

Nothing needs downloading to run the experiment: `--synthetic` is the default.

## Not included

- **No decision-tree implementation.** The repository is named for the
  comparison in the original exercise, and the comparison that matters turned
  out to be between *arrangements of the same data*, not between model families.
  A tree would sit between OLS and 1-NN on the memorisation axis and would not
  change any conclusion here; scikit-learn's is one import away if you want it.
- **No hyperparameter tuning.** Tuning under a leaking arrangement optimises the
  leak.
- **No cross-validation.** Ordinary k-fold on a time series is the shuffled
  split with extra steps. The honest version is a rolling-origin evaluation,
  which is a different piece of machinery and would not change the ranking of
  the two leaks.
- **No claim about Italian air quality.** The honest R² here is a property of
  this feature set and this generator's noise level. The repository measures
  experimental arrangements, not benzene.
- **No imputation experiment.** `KNNImputer` fitted before the split is a fourth
  leak; it is avoided by dropping incomplete rows rather than measured, because
  three leaks already make the point and the fourth needs its own controls.

## License

MIT — see [LICENSE](LICENSE).
