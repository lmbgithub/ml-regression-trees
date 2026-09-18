"""OLS and k-NN, checked against values computed by hand."""

import pytest

from airquality.linalg import LeastSquares, NearestNeighbour, SingularMatrix, solve


def test_solve_recovers_a_known_solution():
    # 2x + y = 5 ; x + 3y = 10  ->  x = 1, y = 3
    assert solve([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]) == pytest.approx([1.0, 3.0])


def test_solve_pivots_around_a_zero_leading_entry():
    # Without partial pivoting this divides by zero.
    assert solve([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0]) == pytest.approx([3.0, 2.0])


def test_a_singular_system_raises_instead_of_returning_nonsense():
    with pytest.raises(SingularMatrix, match="linearly dependent"):
        solve([[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0])


def test_solve_checks_its_shapes():
    with pytest.raises(ValueError, match="square"):
        solve([[1.0, 2.0]], [1.0])


def test_a_perfect_line_is_recovered_exactly():
    model = LeastSquares().fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
    assert model.coefficients[0] == pytest.approx(2.0)
    assert model.intercept == pytest.approx(1.0)


def test_two_features_are_recovered():
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]
    y = [3 * a + 5 * b + 2 for a, b in x]
    model = LeastSquares().fit(x, y)
    assert model.coefficients == pytest.approx([3.0, 5.0])
    assert model.intercept == pytest.approx(2.0)


def test_prediction_matches_the_fit():
    x = [[0.0], [1.0], [2.0], [3.0]]
    model = LeastSquares().fit(x, [1.0, 3.0, 5.0, 7.0])
    assert model.predict([[10.0]])[0] == pytest.approx(21.0)


def test_duplicate_columns_are_reported_not_silently_solved():
    # Exactly the situation the experiment creates on purpose.
    x = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    with pytest.raises(SingularMatrix):
        LeastSquares().fit(x, [1.0, 2.0, 3.0, 4.0])


def test_ridge_makes_a_collinear_fit_solvable():
    x = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    model = LeastSquares(ridge=1e-3).fit(x, [1.0, 2.0, 3.0, 4.0])
    assert model.predict([[5.0, 5.0]])[0] == pytest.approx(5.0, abs=0.01)


def test_a_negative_ridge_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        LeastSquares(ridge=-1.0)


def test_more_parameters_than_rows_is_refused():
    # The fit would be exact and meaningless.
    with pytest.raises(ValueError, match="cannot determine"):
        LeastSquares().fit([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])


def test_fitting_on_nothing_is_refused():
    with pytest.raises(ValueError, match="empty"):
        LeastSquares().fit([], [])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="rows and y"):
        LeastSquares().fit([[1.0], [2.0]], [1.0])


def test_ragged_rows_are_refused():
    with pytest.raises(ValueError, match="same width"):
        LeastSquares().fit([[1.0], [2.0, 3.0], [4.0], [5.0]], [1.0, 2.0, 3.0, 4.0])


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        LeastSquares().predict([[1.0]])


def test_predicting_with_the_wrong_width_is_an_error():
    model = LeastSquares().fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="match the fitted feature count"):
        model.predict([[1.0, 2.0]])


def test_nearest_neighbour_returns_the_training_value():
    model = NearestNeighbour(k=1).fit([[0.0], [10.0]], [1.0, 99.0])
    assert model.predict([[0.1]]) == [1.0]
    assert model.predict([[9.5]]) == [99.0]


def test_nearest_neighbour_memorises_its_training_set():
    # A train R-squared of exactly 1.0 is the signature, and the reason it
    # exposes a shuffled-split leak that a linear model cannot.
    x = [[float(i)] for i in range(10)]
    y = [float(i * i) for i in range(10)]
    model = NearestNeighbour(k=1).fit(x, y)
    assert model.predict(x) == y


def test_k_greater_than_one_averages():
    model = NearestNeighbour(k=2).fit([[0.0], [1.0], [100.0]], [0.0, 10.0, 999.0])
    assert model.predict([[0.4]]) == pytest.approx([5.0])


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        NearestNeighbour(k=0)


def test_k_larger_than_the_training_set_is_refused():
    with pytest.raises(ValueError, match="at least 5"):
        NearestNeighbour(k=5).fit([[0.0], [1.0]], [0.0, 1.0])
