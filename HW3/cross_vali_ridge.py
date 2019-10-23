import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename)
    Y = data['CMEDV']
    X = data.drop('CMEDV', axis = 1)
    X = np.array(X).T
    Y = np.array(Y)
    X_train = X[:,:-46]
    Y_train = Y[:-46]
    X_test = X[:,-46:]
    Y_test = Y[-46:]
    return X_train, Y_train, X_test, Y_test

def sphere(X):
    X_mu = np.mean(X, 1)
    X_std = np.std(X, 1, ddof = 1)
    X_sigma2 = X_std * X_std
    X = (X - X_mu[:, None]) / X_std[:,None]
    return X, X_mu, X_sigma2

def train(X_train, Y_train, lam):
    p, n = X_train.shape[0], X_train.shape[1]
    I_hat = np.eye(p)
    A = np.dot(X_train, X_train.T) + lam*I_hat
    L = np.dot(np.linalg.pinv(A), X_train)
    df = np.trace(np.dot(X_train.T, L))

    X_train = np.r_[np.ones(X_train.shape[1])[None, :], X_train]
    Y_train = Y_train[None, :]
    
    I_hat = np.eye(p+1)
    I_hat[0][0] = 0
    A = np.dot(X_train, X_train.T) + lam*I_hat
    L = np.dot(np.linalg.pinv(A), X_train)
    w = np.dot(L, Y_train.T)
    

    Y_hat = np.dot(w.T, X_train)
    error = mse(Y_hat, Y_train)
    return w, df, error

def mse(Y_hat, Y_test):
    resid = Y_test - Y_hat
    return np.mean(resid*resid)

def test(X_test, Y_test, w):
    X_test = np.r_[np.ones(X_test.shape[1])[None, :], X_test]
    Y_hat = np.dot(w.T, X_test)
    error = mse(Y_hat, Y_test)
    return error

def k_cross_validation(K, lam, X_train_valid, Y_train_valid):
    CV_error = 0
    dk = Y_train_valid.shape[0]//K
    for i in range(K):
        k = dk*i
        X_train = np.c_[X_train_valid[:,:k],X_train_valid[:,k+dk:]]
        X_valid = X_train_valid[:,k:k+dk]
        Y_train = np.r_[Y_train_valid[:k],Y_train_valid[k+dk:]]
        Y_valid = Y_train_valid[k:k+dk]
        w = train(X_train, Y_train, lam)[0]
        e = test(X_valid, Y_valid, w)
        CV_error += e
    CV_error /= K
    print("cv error with lambda = {:.2f} is {:.5f}".format(lam, CV_error,))
    return CV_error

X_train_org, Y_train_org, X_test, Y_test = load_data("boston-corrected")
X_train_valid, X_train_mu, X_train_sigma2 = sphere(X_train_org)
Y_train_valid = Y_train_org
print("X_mu vector is ", X_train_mu)
print("X_sigma vector is ", np.sqrt(X_train_sigma2))

dfs = []
errors = []
lam_list = np.arange(0,20,0.1)
for lam in lam_list:
    w, df, error = train(X_train_valid, Y_train_valid, lam)
    dfs.append(df)
    errors.append(error)
plt.figure()
plt.xlabel("lambda")
plt.ylabel("effective degree of freedom")
plt.plot(lam_list, dfs)
plt.show()

plt.figure()
plt.xlabel("lambda")
plt.ylabel("mean squared error")
plt.plot(lam_list, errors)
plt.show()

w_train = train(X_train_valid, Y_train_valid, 0)[0]
X_test = (X_test - X_train_mu[:,None]) / np.sqrt(X_train_sigma2[:,None]) #sphere(X_test)[0]
e = test(X_test, Y_test, w_train)
print("mse in test set with lambda = 0: {:.3f}".format(e))

lam_star = 0
df_star = 0
cves = []
cve_min = float('inf')
lam_list = np.arange(0,20,0.1)
for lam in lam_list:
    cve = k_cross_validation(10, lam, X_train_valid, Y_train_valid)
    if cve < cve_min:
        cve_min = cve
        lam_star = lam
        df_star = train(X_train_valid, Y_train_valid, lam)[1]
    cves.append(cve)
print("lambda = {:.2f}, df = {:.3f}, CV error = {:.3f}".format(lam_star, df_star, cve_min))

plt.figure()
plt.xlabel("lambda")
plt.ylabel("CV error on test set")
plt.plot(lam_list, cves)
plt.show()

w_ridge = train(X_train_valid, Y_train_valid, lam_star)[0]
error_lam = test(X_test, Y_test, w_ridge)
print("mse in test set with lambda = {:.3f}: {:.10f}".format(lam_star, error_lam))
print("here w = ", w_ridge.T)
print("mse in test set without regularization: {:.10f}".format(e))
print("here w = ", w_train.T)

#for problem 4
w_100 = train(X_train_valid, Y_train_valid, 100)[0]
error_lam = test(X_test, Y_test, w_100)
print("mse in test set with lambda = 100: {:.10f}".format(error_lam))
print("here w = ", w_100.T)