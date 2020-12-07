import numpy as np
import pandas as pd
import pytest

from bin.RawDataUtilities import RawDataUtilities


@pytest.fixture
def data():
    """Inject data class to each test cases"""
    return pd.read_csv("../data/ex1data2.txt", names=['Square', 'Bedrooms', 'Price'])


def test_normalize_feature():
    x_training_mat = np.asmatrix([[1., -1., 2.],
                                  [2., 0., 0.],
                                  [0., 1., -1.]])

    x_expected_mat = np.asmatrix([[0., -1.22, 1.33],
                                  [1.22, 0., -0.26],
                                  [-1.22, 1.22, -1.06]])

    np.testing.assert_allclose(RawDataUtilities.normalize_feature(x_training_mat), x_expected_mat, rtol=1e-1)


def test_normalize_feature_df(data):
    training_mat = np.asmatrix(data)
    data = RawDataUtilities.normalize_feature_data_frame(data)
    df = RawDataUtilities.normalize_feature(training_mat)

    np.testing.assert_allclose(data, df, rtol=1e-1)


def test_polynomial_features():
    x_training_mat = np.asmatrix([[1, 2, 3]])

    # 0
    # x1 + x2 + x3
    # x1 * x1 + x1 * x2 + x1 * x3
    # x2 * x2 + x2 * x3
    # x3 * x3
    np.testing.assert_array_equal(np.asmatrix([[1., 1., 2., 3., 1., 2., 3., 4., 6., 9.]]),
                                  RawDataUtilities.polynomial_features(x_training_mat, 2))
