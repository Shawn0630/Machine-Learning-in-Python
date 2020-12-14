import numpy as np
from bin.Regularization import Regularization
import matplotlib.pyplot as plt


# https://juejin.cn/post/6844903623109902349
# https://zhuanlan.zhihu.com/p/45190898
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='logistic', la=0):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative

        self.weights = []
        self.regularization = Regularization(la)
        for i in range(1, len(layers) - 1):
            # Return random floats in the half-open interval [-0.25, 0.25]
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, x_train_mat, y_train_mat, learning_rate=0.02, epochs=20000):
        """ gradient for linear regression
           Parameters
           ----------
           x_train_mat: matrix
           y_train_mat: array
           learning_rate: matrix in n * 1
               n features
           epochs:
        """
        x_train_mat = NeuralNetwork.append_ones_at_last(x_train_mat)
        self._gradient_descent(x_train_mat, y_train_mat, learning_rate, epochs)

    def predict(self, x_predict_mat):
        x_predict_mat = NeuralNetwork.append_ones_at_last(x_predict_mat)
        a = NeuralNetwork.feed_forward(self.weights, x_predict_mat, self.activation)

        return a[-1]

    @staticmethod
    def cost_function(weights, x_train_mat, y_train_mat, activation):
        a = NeuralNetwork.feed_forward(weights, x_train_mat, activation)

        return np.dot((y_train_mat - a[-1]).T, (y_train_mat - a[-1]))[0, 0]

    @staticmethod
    def feed_forward(weights, x_train_mat, activation):
        a = [x_train_mat]

        for l in range(len(weights)):
            a.append(activation(np.dot(a[l], weights[l])))

        return a

    @staticmethod
    def deltas(weights, a, m, y_train_mat, activation_deriv, regularization):
        error = y_train_mat - a[-1]
        deltas = [error * activation_deriv(a[-1])]
        for l in range(len(a) - 2, 0, -1):
            gradient = np.multiply(deltas[-1].dot(weights[l].T), activation_deriv(a[l]))
            regular_term = regularization.regularization_term_gradient(m, weights[l])
            deltas.append(gradient + regular_term.T)

        deltas.reverse()

        return deltas

    def _gradient_descent(self, x_train_mat, y_train_mat, learning_rate=0.2, epochs=20000):
        m = x_train_mat.shape[0]
        loss_list = []
        epochs_list = []
        for k in range(epochs):  # 循环epochs次
            i = np.random.randint(x_train_mat.shape[0])
            a = NeuralNetwork.feed_forward(self.weights, x_train_mat[i], self.activation)
            cost = NeuralNetwork.cost_function(self.weights, x_train_mat, y_train_mat, self.activation)
            loss_list.append(cost)
            epochs_list.append(k)
            deltas = NeuralNetwork.deltas(self.weights, a, m, y_train_mat[i],
                                          self.activation_deriv, self.regularization)

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

        plt.plot(epochs_list, loss_list)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

    @staticmethod
    def append_ones_at_last(x_train_mat):
        temp = np.ones([x_train_mat.shape[0], x_train_mat.shape[1] + 1])
        temp[:, 0:-1] = x_train_mat
        return temp
