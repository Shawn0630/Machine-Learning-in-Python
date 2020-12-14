import numpy as np
import cvxopt
import cvxopt.solvers

sigma = 10.0


def kernel(xi, xj):  # 高斯核函数（数据集线性不可分）
    M = xi.shape[0]
    K = np.zeros((M, 1))
    for l in range(M):
        A = np.array([xi[l]]) - xj
        K[l] = [np.exp(-0.5 * float(A.dot(A.T)) / (sigma ** 2))]
    return K


def polynomial_kernel(x, y, C=1, d=3):
    # Inputs:
    #   x   : vector of x data.
    #   y   : vector of y data.
    #   c   : is a constant
    #   d   : is the order of the polynomial.
    return (np.dot(x, y) + C) ** d


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gaussian_kernel(x, y, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


class SVM:
    def __init__(self, kernel='linear', C=0, gamma=1, degree=3):

        if C is None:
            C = 0
        if gamma is None:
            gamma = 1
        if kernel is None:
            kernel = 'linear'

        C = float(C)
        gamma = float(gamma)
        degree = int(degree)

        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):

                # Kernel trick.
                if self.kernel == 'linear':
                    K[i, j] = linear_kernel(X[i], X[j])
                if self.kernel == 'gaussian':
                    K[i, j] = gaussian_kernel(X[i], X[j], self.gamma)  # Kernel trick.
                    self.C = None  # Not used in gaussian kernel.
                if self.kernel == 'polynomial':
                    K[i, j] = polynomial_kernel(X[i], X[j], self.C, self.degree)

        # Converting into cvxopt format:
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None or self.C == 0:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Restricting the optimisation with parameter C.
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve QP problem:
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alphas = np.ravel(solution['x'])  # Flatten the matrix into a vector of all the Langrangian multipliers.
