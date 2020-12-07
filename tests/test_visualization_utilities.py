import numpy as np
from bin.VisualizatinonUtilities import VisualizationUtilities


def test_data_generator_3d():
    x_arr = np.linspace(-5, 5, 10)
    y_arr = np.linspace(-5, 5, 10)
    x, y = VisualizationUtilities.data_generator_3d(x_arr, y_arr)

    print(x_arr)
    print(y_arr)
    print(x)
    print(y)
