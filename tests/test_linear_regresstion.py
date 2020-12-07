import pytest
import numpy as np
import numpy.testing as np_test
import matplotlib.pyplot as plt
from bin.LinearRegression import LinearRegression
from bin.IOUtilities import IOUtilities
from bin.VisualizatinonUtilities import VisualizationUtilities


@pytest.fixture
def linear_regression():
    """Inject LinearRegression class to each test cases"""
    return LinearRegression()


@pytest.fixture
def data():
    """Inject data class to each test cases"""
    return IOUtilities.read_data("../data/ex1data1.txt", ['Populations', 'Profit'])


def test_cost_function_all_fit(linear_regression):
    m = 100
    n = 4
    x_mat = np.asmatrix(np.arange(m * n).reshape(m, n), dtype="float64")
    theta = np.asmatrix(np.arange(1 * n).reshape(1, n), dtype="float64")
    y_mat = np.dot(x_mat, theta.T)

    cost = linear_regression.cost_function(theta, x_mat, y_mat)
    assert cost == 0.0, "cost should equals 0 when all samples fit"


def test_cost_function_not_fit(linear_regression):
    m = 100
    n = 4
    x_mat = np.asmatrix(np.arange(m * n).reshape(m, n), dtype="float64")
    theta = np.asmatrix(np.arange(1 * n).reshape(1, n), dtype="float64")
    y_mat = np.dot(x_mat, theta.T) + np.asmatrix(np.ones((m, 1)))

    cost = linear_regression.cost_function(theta, x_mat, y_mat)
    assert cost == 0.5, "cost is not correct"


def test_gradient_all_fit(linear_regression):
    m = 100
    n = 4
    x_mat = np.asmatrix(np.arange(m * n).reshape(m, n), dtype="float64")
    theta = np.asmatrix(np.arange(1 * n).reshape(1, n), dtype="float64")
    y_mat = np.dot(x_mat, theta.T)

    gradient = linear_regression.gradient(theta, x_mat, y_mat)
    expected = np.asmatrix(np.zeros((1, 4)))
    np_test.assert_array_equal(gradient, expected, "gradient should all equals 0 when all samples fit")


def test_gradient_not_fit(linear_regression):
    m = 100
    n = 4
    x_mat = np.asmatrix(np.arange(m * n).reshape(m, n), dtype="float64")
    theta = np.asmatrix(np.arange(1 * n).reshape(1, n), dtype="float64")
    y_mat = np.dot(x_mat, theta.T) + np.asmatrix(np.ones((m, 1)))

    gradient = linear_regression.gradient(theta, x_mat, y_mat)
    expected = np.asmatrix([-198., -199., -200., -201.])
    np_test.assert_array_equal(gradient, expected, "gradient should equals 0 when all samples fit")


def test_linear_regression(data, linear_regression):
    x_mat = data[0]
    y_mat = data[1]

    model = linear_regression.fit(x_mat, y_mat)
    print(model.predict(x_mat[21, :]))


def test_plot_cost_method(data, linear_regression):
    x_mat = data[0]
    y_mat = data[1]

    x_mat = np.hstack([np.ones((len(x_mat), 1)), x_mat])

    def cost_function(theta):
        return linear_regression.cost_function(theta, x_mat, y_mat)

    # def cost_function(theta):
    #     return np.sin(np.sqrt(theta[:, 0] ** 2 + theta[:, 1] ** 2))

    VisualizationUtilities.plot_3d(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25), cost_function)
    plt.show()
