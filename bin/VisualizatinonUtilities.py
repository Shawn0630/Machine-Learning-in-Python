import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itertools


class VisualizationUtilities:
    markers = itertools.cycle((',', '+', '.', 'o', '*', 's', 'x'))
    colors = itertools.cycle(('r', 'g', 'b', 'c', 'm', 'y', 'k'))

    @staticmethod
    def data_generator(low, high, num):
        return np.linspace(low, high, num)

    @staticmethod
    def data_generator_3d(x_arr, y_arr):
        return np.meshgrid(x_arr, y_arr)

    @staticmethod
    def plot_3d(x_range_arr, y_range_arr, function):

        # Preparing data
        z = np.zeros(shape=(len(x_range_arr), len(y_range_arr)))
        for i in range(len(x_range_arr)):
            for j in range(len(y_range_arr)):
                z[i][j] = function(np.asmatrix([x_range_arr[i], y_range_arr[j]]))
        x_range_arr, y_range_arr = np.meshgrid(x_range_arr, y_range_arr)

        # fig = plt.gcf()
        # ax = fig.gca(projection='3d')
        # ax.contour3D(x_range_arr, y_range_arr, z, 100, cmap='binary')
        # plt.show()

        fig = plt.gcf()
        ax = fig.gca(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(x_range_arr, y_range_arr, z, cmap='viridis',
                               linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()

    @staticmethod
    def plot_scatter_2d(x_train_mat, classifier_index_tup_list):
        fig = plt.gcf()
        ax = fig.gca()

        for t in classifier_index_tup_list:
            label = t[0]
            index = t[1]
            ax.scatter(np.asarray(x_train_mat[index[0], 0]), np.asarray(x_train_mat[index[0], 1]), marker=next(VisualizationUtilities.markers),
                       label=label, c=next(VisualizationUtilities.colors))

        ax.legend()

    # referring to https://zhuanlan.zhihu.com/p/52113487
    @staticmethod
    def plot_decision_boundary_2d(x_train_mat, predict):

        # Preparing data
        x_min = np.min(x_train_mat[:, 0])
        x_max = np.max(x_train_mat[:, 0])
        y_min = np.min(x_train_mat[:, 1])
        y_max = np.max(x_train_mat[:, 1])

        xs = np.arange(x_min, x_max, 0.05)
        ys = np.arange(y_min, y_max, 0.05)
        x, y = np.meshgrid(xs, ys)

        x_pred_mat = np.c_[x.ravel(), y.ravel()]
        y_pred_mat = predict(x_pred_mat)
        y_pred_mat = y_pred_mat.reshape(x.shape)

        fig = plt.gcf()
        plt.contourf(x, y, y_pred_mat, cmap=plt.cm.brg, alpha=0.2)


    @staticmethod
    def plot_mat(x_train_array, size):
        x_size = size[0]
        y_size = size[1]

        fig = plt.gcf()
        ax = fig.gca()
        ax.matshow(x_train_array.reshape((x_size, y_size)), cmap=cm.binary)
        plt.xticks(np.array([]))  # just get rid of ticks
        plt.yticks(np.array([]))





