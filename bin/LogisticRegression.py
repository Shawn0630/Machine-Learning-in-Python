import numpy as np
from sklearn.metrics import accuracy_score
from bin.GradientDescent import GradientDescent
import scipy.optimize as opt
from bin.RawDataUtilities import RawDataUtilities


class LogisticRegression:

    def __init__(self, degree=1, la=0):
        self.coef_ = None
        self.intercept_ = None
        self.theta = None
        self._degree = degree
        self._la = la

    @staticmethod
    def _sigmiod(t):
        return 1. / (1. + np.exp(-t))

    # def fit(self, x_train_mat, y_train_mat, eta=0.01, n_iters=1e4):
    #     assert x_train_mat.shape[0] == y_train_mat.shape[0], \
    #         "the size of X_train must be equal to the size of y_train"
    #
    #     x_train_mat = np.hstack([np.ones((len(x_train_mat), 1)), x_train_mat])
    #     self.theta = GradientDescent.batch_gradient_descent(x_train_mat, y_train_mat, self.cost_function,
    #                                                         self.gradient, max_epochs=n_iters, epsilon=eta,
    #                                                         alpha=0.0001)
    #     self.intercept_ = self.theta[0, 0]
    #     self.coef_ = self.theta[0, 1:]
    #     return self

    def fit(self, x_train_mat, y_train_mat):
        assert x_train_mat.shape[0] == y_train_mat.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        x_train_mat = RawDataUtilities.polynomial_features(x_train_mat, self._degree)[:, 1:]

        x_train_mat = np.hstack([np.ones((len(x_train_mat), 1)), x_train_mat])
        theta = np.zeros((1, x_train_mat.shape[1]))
        # self.theta = opt.fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,
        #                            args=(x_train_mat, y_train_mat))[0]
        self.theta = opt.minimize(fun=self.cost_function, x0=theta, args=(x_train_mat, y_train_mat),
                                  method='TNC', jac=self.gradient).x
        # self.intercept_ = self.theta[0, 0]
        # self.coef_ = self.theta[0, 1:]
        return self

    def predict_proda(self, x_predict):
        # assert self.intercept_ is not None and self.coef_ is not None, \
        #     "must fit before predict!"
        # assert x_predict.shape[1] == len(self.coef_), \
        #     "the feature number of X_predict must be equal to X_train"

        x_predict = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        return self._sigmiod(x_predict.dot(self.theta))

    def predict(self, x_predict):
        # assert self.intercept_ is not None and self.coef_ is not None, \
        #     "must fit before predict!"
        # assert x_predict.shape[1] == len(self.coef_), \
        #     "the feature number of X_predict must be equal to X_train"

        x_predict = RawDataUtilities.polynomial_features(x_predict, self._degree)[:, 1:]

        proda = self.predict_proda(x_predict)
        return np.array(proda >= 0.5, dtype='int')

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)

        return accuracy_score(y_test, y_predict)

    @staticmethod
    def cost_function(theta, x_training_mat, y_training_mat, la=0):
        theta = np.reshape(theta, (x_training_mat.shape[1], 1)) # theta, n * 1, n features
        sig = LogisticRegression._sigmiod(np.dot(x_training_mat, theta))
        m = x_training_mat.shape[0]
        cost = (np.dot(-y_training_mat.T, np.log(sig)) - np.dot(1 - y_training_mat.T, np.log(1 - sig))) / m
        cost = cost + np.dot(theta.T[0, 1:], theta[1:, 0]) * la / (2 * m)  # skip theta_0 as convention
        return cost

    @staticmethod
    def gradient(theta, x_training_mat, y_training_mat, la=0):  # theta was converted to an array
        theta = np.reshape(theta, (x_training_mat.shape[1], 1))
        m = x_training_mat.shape[0]
        grad = (np.dot(x_training_mat.T, LogisticRegression._sigmiod(x_training_mat.dot(theta)) - y_training_mat)) / m
        regularized_term = (float(la) / m) * np.asmatrix(theta[1:, 0])  # theta[1:, 0] -> array np.asmatrix(theta[1:, 0]) -> [[0, 0]]
        regularized_term = np.concatenate([np.asmatrix([0]), regularized_term.T])

        return grad + regularized_term

    def __repr__(self):
        return "LogisticRegression()"
