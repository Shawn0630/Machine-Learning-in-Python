import pandas as pd
import numpy as np


class IOUtilities:

    @staticmethod
    def read_data(file_path, names):
        """read data a file

        Parameters
        ----------
        file_path: str
            path to the file
        names: list
            a list of header

        Returns
        -------
        x_mat: np.matrix
            inputs in matrix
        y_mat: np.matrix
            expected output
        """

        data = pd.read_csv(file_path, header=None, names=names)
        cols = data.shape[1]
        x = data.iloc[:, 0:cols - 1]
        y = data.iloc[:, cols - 1:cols]

        x_mat = np.asmatrix(x.values)
        y_mat = np.asmatrix(y.values)

        return x_mat, y_mat
