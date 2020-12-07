import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:

    @staticmethod
    def batch_gradient_descent(data_x, data_y, cost_func, gradient_func, alpha=0.01, max_epochs=10000, epsilon=1e-4):
        """Gradient decent in batch manner, returns weight and its cost

        Parameters
        ----------
        data_x: matrix
           The training dataset
        data_y: matrix
           The expected output of training dataset
        cost_func: callable ``cost_func(x, *args)``
           The number of legs the animal (default is 4)
        gradient_func: callable ``gradient_func()``
        weights
        alpha
        max_epochs
        epsilon

        Returns
        -------
        weight: matrix in 1 * n
            the weight
        """
        x_mat = np.mat(data_x)
        y_mat = np.mat(data_y)
        m, n = x_mat.shape
        weights = np.ones((1, n))                          # Initial weights
        epochs_count = 0
        cost_list = []
        epochs_list = []
        while epochs_count < max_epochs:
            cost = cost_func(weights, x_mat, y_mat)        # Loss of weights from last round
            grad = gradient_func(weights, x_mat, y_mat)    # Gradient of weights from last round
            weights = weights - alpha * grad               # Update weights
            loss_new = cost_func(weights, x_mat, y_mat)    # New cost of weights from current round
            if abs(loss_new - cost) < epsilon:             # terminate condition
                break
            cost_list.append(loss_new)
            epochs_list.append(epochs_count)
            epochs_count += 1
        plt.plot(epochs_list, cost_list)
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.show()
        print(weights)
        return weights

    @staticmethod
    def stochastic_gradient_descent(data_x, data_y, cost_func, gradient_func, alpha=0.1, max_epochs=10000, epsilon=1e-4):
        """Gradient decent in stochastic manner, returns weight and its cost

        Parameters
        ----------
        data_x : matrix
           The input
        data_y : matrix
           The expected output
        cost_func : int, optional
           The number of legs the animal (default is 4)
        gradient_func :
        weights
        alpha
        max_epochs
        epsilon
        """
        x_mat = np.mat(data_x)
        y_mat = np.mat(data_y)
        m, n = x_mat.shape
        weights = np.ones((1, n))                          # Initial weights
        epochs_count = 0
        loss_list = []
        epochs_list = []
        while epochs_count < max_epochs:
            rand_i = np.random.randint(m)                           # Random selected sample
            loss = cost_func(weights, x_mat, y_mat)                 # Loss of weights from last round
            grad = gradient_func(weights, x_mat[rand_i, :], y_mat[rand_i, :])  # Gradient of weights from last round
            weights = weights - alpha * grad                        # Update weights
            loss_new = cost_func(weights, x_mat, y_mat)             # New loss of weights from current round
            if abs(loss_new - loss) < epsilon:
                break
            loss_list.append(loss_new)
            epochs_list.append(epochs_count)
            epochs_count += 1
        plt.plot(epochs_list, loss_list)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        return weights
