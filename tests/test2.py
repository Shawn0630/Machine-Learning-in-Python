from bin.NeuralNetwork import NeuralNetwork
from bin.VisualizatinonUtilities import VisualizationUtilities

import scipy.io as sio
import numpy as np

from sklearn.metrics import classification_report  # 这个包是评价报告


def load_data(path, transpose=True):
    data = sio.loadmat(path)

    y_train_array = data.get('y')
    y_train_array = y_train_array.reshape(y_train_array.shape[0])

    x_train_array = data.get('X')
    if transpose:
        x_train_list = np.asarray([im.reshape((20, 20)).T for im in x_train_array])  # [] - list
    else:
        x_train_list = np.asarray([im.reshape(400) for im in x_train_array])

    return x_train_list, y_train_array


x_raw, y_raw = load_data('../data/ex3data1.mat', False)

nn = NeuralNetwork([400, 25, 10], activation='tanh')

y_matrix = []

for k in range(1, 11):
    y_matrix.append((y_raw == k).astype(int))  # 见配图 "向量化标签.png"

# last one is k==10, it's digit 0, bring it to the first position，最后一列k=10，都是0，把最后一列放到第一列
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y_matrix = np.asarray(y_matrix).T

nn.fit(np.asmatrix(x_raw), y_matrix)

y_predict = nn.predict(np.asmatrix(x_raw))
y_pred = np.argmax(y_predict, axis=1)

y_raw[y_raw == 10] = 0

print(classification_report(y_raw, y_pred))
