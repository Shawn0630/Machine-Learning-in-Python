from bin.NeuralNetwork import NeuralNetwork
from bin.IOUtilities import IOUtilities
import pytest
import numpy as np


@pytest.fixture
def data():
    """Inject data class to each test cases"""
    return IOUtilities.read_data("../data/xor.txt", ['x1', 'x2', 'Output'])


def test_xor(data):
    x_train_mat = data[0]
    y_train_mat = data[1]
    nn = NeuralNetwork([2, 2, 1], 'tanh')
    nn.fit(x_train_mat, y_train_mat)
    test_cases = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    for test_case in test_cases:
        np.testing.assert_almost_equal(test_case[1], nn.predict(np.asmatrix(test_case[0])), decimal=1)
