import numpy as np
from sklearn.metrics import accuracy_score
from .GradientDescent import GradientDescent


class LinearRegression:

    def __init__(self):
        """Constructor"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit(self, x_train_mat, y_train_mat, eta=1e-20, n_iters=1e4):
        """Train a Linear Regression model from training data x_train_mat, y_train_mat by using gradient descent

        Parameters
        ----------
        x_train_mat: training data in the format of matrix of size m * n
            m samples, n features
        y_train_mat: traning data in the format of matrix of size m * 1
            m samples
        eta:
            n features
        n_iters:

        """

        assert x_train_mat.shape[0] == y_train_mat.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        x_train_mat = np.hstack([np.ones((len(x_train_mat), 1)), x_train_mat])
        self._theta = GradientDescent.batch_gradient_descent(x_train_mat, y_train_mat, self.cost_function,
                                                             self.gradient, max_epochs=n_iters, epsilon=eta, alpha=0.01)
        self.intercept_ = self._theta[0, 0]
        self.coef_ = self._theta[0, 1:]

        return self

    def predict(self, x_predict):
        """Return prediction for an input

        Parameters
        ----------
        x_predict: matrix in 1 * n
           input data

        Returns
        -------
        y_predict: float
            the prediction
        """
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert x_predict.shape[1] == len(self.coef_), \
            "the feature number of x_predict must be equal to x_train"

        x_predict = np.hstack([np.ones((len(x_predict), 1)), x_predict])

        return np.dot(x_predict, self._theta.T)[0, 0]

    @staticmethod
    def cost_function(theta, x_mat, y_mat):
        """ cost function for linear regression

        Parameters
        ----------
        x_mat: matrix in m * n
            m samples, n features
        y_mat: matrix in m * 1
            m samples
        theta : matrix in n * 1
            n features

        Returns
        -------
        cost: float
            the error
        """
        m = x_mat.shape[0]

        inner = np.dot(x_mat, theta.T) - y_mat

        square_sum = np.dot(inner.T, inner)
        cost = square_sum / (2 * m)

        return cost[0, 0]

    @staticmethod
    def gradient(theta, x_mat, y_mat):
        """ gradient for linear regression

        Parameters
        ----------
        x_mat: matrix in m * n
            m samples, n features
        y_mat: matrix in m * 1
            m samples
        theta : matrix in n * 1
            n features

        """
        m = x_mat.shape[0]
        inner = np.dot((np.dot(x_mat, theta.T) - y_mat).T, x_mat)

        return inner / m

    def score(self, x_test_mat, y_test_mat):
        y_predict = self.predict(x_test_mat)

        return accuracy_score(y_test_mat, y_predict)

    @staticmethod
    def normal_equation(x_mat, y_mat):
        x_mat = np.hstack([np.ones((len(x_mat), 1)), x_mat])

        return np.linalg.inv(np.dot(x_mat.T, x_mat)) * x_mat.T * y_mat
