import pytest
import numpy as np
from bin.IOUtilities import IOUtilities
from bin.LogisticRegression import LogisticRegression
from bin.RawDataUtilities import RawDataUtilities
from bin.VisualizatinonUtilities import VisualizationUtilities
import matplotlib.pyplot as plt


@pytest.fixture
def data():
    """Inject data class to each test cases"""
    return IOUtilities.read_data("../data/ex2data1.txt", names=['Exam 1', 'Exam 2', 'Admitted'])

@pytest.fixture
def data2():
    """Inject data class to each test cases"""
    return IOUtilities.read_data("../data/ex2data2.txt", names=['Test 1', 'Test 2', 'Accepted'])


def test_logistic_regression(data):
    x_train_mat = data[0]
    y_train_mat = data[1]

    logistic_regression = LogisticRegression()
    model = logistic_regression.fit(x_train_mat, y_train_mat)
    x_train_mat = np.hstack([np.ones((len(x_train_mat), 1)), x_train_mat])
    print(logistic_regression.cost_function(np.zeros((1, x_train_mat.shape[1])), x_train_mat, y_train_mat))
    print(logistic_regression.cost_function(model.theta[0, :], x_train_mat, y_train_mat))


def test_logistic_regression(data):
    x_train_mat = data[0]
    y_train_mat = data[1]

    logistic_regression = LogisticRegression()
    logistic_regression.fit(x_train_mat, y_train_mat)

    positive = ('positive', np.where(y_train_mat > 0.5))
    negative = ('negative', np.where(y_train_mat < 0.5))
    classifier_label_list = [positive, negative]
    VisualizationUtilities.plot_scatter_2d(x_train_mat, classifier_label_list)
    VisualizationUtilities.plot_decision_boundary_2d(x_train_mat, logistic_regression.predict)
    plt.show()


def test_gradient_function(data):
    x_train_mat = data[0]
    y_train_mat = data[1]
    x_train_mat = np.hstack([np.ones((len(x_train_mat), 1)), x_train_mat])
    theta = np.ones((1, x_train_mat.shape[1]))

    np.testing.assert_allclose(np.asmatrix([[0.4], [20.81292044], [21.84815684]]),
                               LogisticRegression.gradient(theta, x_train_mat, y_train_mat))


def test_cost_function(data):
    x_train_mat = data[0]
    y_train_mat = data[1]
    x_train_mat = np.hstack([np.ones((len(x_train_mat), 1)), x_train_mat])
    theta = np.zeros((1, x_train_mat.shape[1]))

    np.testing.assert_almost_equal(0.69314718, LogisticRegression.cost_function(theta, x_train_mat, y_train_mat))


def test_plot_scatter(data):
    x_train_mat = data[0]
    y_train_mat = data[1]

    positive = ('positive', np.where(y_train_mat > 0.5))
    negative = ('negative', np.where(y_train_mat < 0.5))
    classifier_label_list = [positive, negative]
    VisualizationUtilities.plot_scatter_2d(x_train_mat, classifier_label_list)
    plt.show()


def test_logistic_regression_with_reg(data2):
    x_train_mat = data2[0]
    y_train_mat = data2[1]

    logistic_regression = LogisticRegression(degree=8, la=100)
    logistic_regression.fit(x_train_mat, y_train_mat)

    positive = ('positive', np.where(y_train_mat > 0.5))
    negative = ('negative', np.where(y_train_mat < 0.5))
    classifier_label_list = [positive, negative]
    VisualizationUtilities.plot_scatter_2d(x_train_mat, classifier_label_list)
    VisualizationUtilities.plot_decision_boundary_2d(x_train_mat, logistic_regression.predict)

    plt.show()

    print(logistic_regression.score(x_train_mat, y_train_mat))
