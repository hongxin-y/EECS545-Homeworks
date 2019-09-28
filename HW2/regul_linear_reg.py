import scipy.io as scio
import numpy as np
import pandas as pd

data = scio.loadmat("bodyfat_data.mat")
X, Y = data.get('X'). data, data.get('y'). data
print(np.ones(150).shape, np.transpose(X[:150]).shape)
X_train, Y_train = np.r_[[np.ones(150)], np.transpose(X[:150])], Y[:150]
X_test, Y_test = np.r_[[np.ones(len(X)-150)], np.transpose(X[150:])], Y[150:]

n, p = X.shape[0], X.shape[1]
lam = 10
I_hat = np.eye(p+1)
I_hat[0][0] = 0
print(I_hat)
A = np.dot(X_train, np.transpose(X_train)) - 2*lam*I_hat

C = np.dot(X_train, Y_train)
w = np.dot(np.linalg.inv(A), C)
print(w)
res_train = np.dot(np.transpose(X_train),w) - Y_train
res = np.dot(np.transpose(X_test), w) - Y_test

print(res)
print(res_train)
print(np.mean(res_train*res_train))
print(np.mean(res*res))

x_pre_T = np.array([1,100,100])
y_pre = np.dot(x_pre_T, w)
print(y_pre)
