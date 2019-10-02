import scipy.io as scio
import numpy as np
import pandas as pd

def load_data(filename):
    data = scio.loadmat(filename)
    X, Y = data.get('X'). data, data.get('y'). data
    print(np.ones(150).shape, np.transpose(X[:150]).shape)
    X_train, Y_train = np.r_[[np.ones(150)], np.transpose(X[:150])], Y[:150]
    X_test, Y_test = np.r_[[np.ones(len(X)-150)], np.transpose(X[150:])], Y[150:]
    return X_train, Y_train, X_test, Y_test

def train(X_train, Y_train, lam):
    p, n = X_train.shape[0]-1, X_train.shape[1]
    I_hat = np.eye(p+1)
    I_hat[0][0] = 0
    A = np.dot(X_train, np.transpose(X_train)) + lam*I_hat
    C = np.dot(X_train, Y_train)
    w = np.dot(np.linalg.inv(A), C)
    return w

X_train, Y_train, X_test, Y_test = load_data("bodyfat_data.mat")
w = train(X_train, Y_train, 10)

print("parameters w = ", w)
res = np.dot(np.transpose(X_test), w) - Y_test
mse = np.mean(np.dot(res[0], res[0]))

print("mean squared error on testset = %.2f"%mse)

x_pre_T = np.array([1,100,100])
y_pre = np.dot(x_pre_T, w)
print("predicted y on X =", x_pre_T[1:], "is", y_pre[0])
