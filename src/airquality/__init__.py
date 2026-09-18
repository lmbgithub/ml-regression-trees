"""Three ways to get R-squared near 1.0 on UCI Air Quality, and the honest number.

Standard library only: the regression, the metrics and the synthetic series are
all written out, so the whole experiment runs with no download and no
scikit-learn. `data.load_uci` and the tree models are the only parts that need
anything installed.
"""

from airquality.data import Dataset, Row, load_csv, load_uci, parse_csv
from airquality.experiment import Arrangement, run, table, verdict
from airquality.linalg import LeastSquares, NearestNeighbour
from airquality.metrics import Scores, mae, r2, rmse, score
from airquality.split import chronological, shuffled

__all__ = [
    "Arrangement",
    "Dataset",
    "LeastSquares",
    "NearestNeighbour",
    "Row",
    "Scores",
    "chronological",
    "load_csv",
    "load_uci",
    "mae",
    "parse_csv",
    "r2",
    "rmse",
    "run",
    "score",
    "shuffled",
    "table",
    "verdict",
]
