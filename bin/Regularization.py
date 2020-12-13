import numpy as np


class Regularization:
    def __init__(self, la=0):
        self.la = la

    def regularization_term(self, m, theta):
        return np.dot(theta.T[0, 1:], theta[1:, 0]) * self.la / (2 * m)  # skip theta_0 as convention

    def regularization_term_gradient(self, m, theta):
        regularized_term = (float(self.la) / m) * np.asmatrix(
            theta[1:, 0])  # theta[1:, 0] -> array np.asmatrix(theta[1:, 0]) -> [[0, 0]]
        regularized_term = np.concatenate([np.asmatrix([0]), regularized_term.T])

        return regularized_term
