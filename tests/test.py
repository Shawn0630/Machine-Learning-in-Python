import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def costFunction(theta, X, y, lamda):
    theta = np.reshape(theta, (X.shape[1], 1))
    sig = sigmoid(np.dot(X, theta))
    m = X.shape[0]
    cost = (np.dot(-y.T, np.log(sig)) - np.dot(1 - y.T, np.log(1 - sig))) / m
    cost = cost + np.dot(theta.T[0, 1:], theta[1:, 0]) * lamda / (2 * m)
    return cost


def gradient(theta, X, y, lamda):
    theta = np.reshape(theta, (X.shape[1], 1))
    m = X.shape[0]
    sig = sigmoid(np.dot(X, theta))
    theta[0] = 0
    grad = np.zeros([X.shape[1], 1])
    grad = np.dot(X.T, (sig - y)) / m
    grad = grad + theta * lamda / m
    return grad


def plotDecisionBoundary(theta, X, y):
    x1_min = np.min(X[:, 1])
    x1_max = np.max(X[:, 1])
    x1 = np.arange(x1_min, x1_max, 0.5)
    x2 = -(theta[0] + theta[1] * x1) / theta[2]
    plt.plot(x1, x2, '-')
    plt.legend(['decision boundary', 'Admitted', 'not Admitted'], loc='upper right')

    plt.show()


def plotdata(X, y):
    postive = np.where(y > 0.5)
    negtive = np.where(y < 0.5)
    plt.scatter(X[postive[0], 0], X[postive[0], 1], marker='o', c='g')
    plt.scatter(X[negtive[0], 0], X[negtive[0], 1], marker='x', c='r')


# Part 1: Plotting
data = np.loadtxt('../data/ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2:3]
print('X:', X.shape)
print('y:', y.shape)

plotdata(X, y)
plt.xlabel('exam 1 score')
plt.ylabel('exam 2 score')
plt.legend(['Admitted', 'not Admitted'], loc='upper right')

# Part 2: Compute Cost and Gradient
[m, n] = X.shape
ones = np.ones((m, 1))
X = np.column_stack((ones, X))
initial_theta = np.zeros((n + 1, 1))
lamda = 0
cost = costFunction(initial_theta, X, y, lamda)
grad = gradient(initial_theta, X, y, lamda)
print("Cost at initial theta (zeros):", cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):\n', grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')

test_theta = [[-24], [0.2], [0.2]]
cost = costFunction(test_theta, X, y, lamda)
grad = gradient(test_theta, X, y, lamda)
print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:\n', grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')
# Part 3: Optimizing using fminunc

result = opt.minimize(fun=costFunction,
                      x0=initial_theta,
                      args=(X, y, lamda),
                      method='TNC',
                      jac=gradient)
print('Cost at theta found by fminunc:', result.fun)
print('Expected cost (approx): 0.203')
print('theta:', result.x)
print('Expected theta (approx):')
print('-25.161\n 0.206\n 0.201')
# Plot Boundary
theta = result.x
plotDecisionBoundary(theta, X, y)
plt.legend(['decision boundary', 'Admitted', 'not Admitted'], loc='upper right')
# Part 4: Predict and Accuracies
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85, we predict an admission')
print('probability of', prob)

h = sigmoid(np.dot(X, theta))
index = np.where(h >= 0.5)
p = np.zeros([m, 1])
p[index] = 1
prob = np.mean(np.double(p == y)) * 100
print('Train Accuracy:', prob)
print('Expected accuracy (approx): 89.0')