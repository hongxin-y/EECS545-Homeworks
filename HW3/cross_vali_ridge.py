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
    n = X.shape[1]
    X_mu = np.mean(X, 1)
    X_sigma2 = (np.mean(X*X, 1) - (X_mu*X_mu))*(n/(n-1))
    X = (X - X_mu[:, None]) / np.sqrt(X_sigma2[:,None])
    return X, X_mu, X_sigma2

def train(X_train, Y_train, lam):
    X_train = np.r_[np.array([np.ones(X_train.shape[1])]), X_train]
    Y_train = np.array([Y_train])
    p, n = X_train.shape[0] - 1, X_train.shape[1]
    I_hat = np.eye(p+1)
    I_hat[0][0] = 0
    A = np.dot(X_train, X_train.T) + lam*I_hat
    L = np.dot(np.linalg.inv(A), X_train)
    w = np.dot(L, Y_train.T)
    df = np.trace(np.dot(X_train.T, L))

    Y_hat = np.dot(w.T, X_train)
    error = mse(Y_hat, Y_train)
    return w, df, error

def mse(Y_hat, Y_train):
    resid = Y_train - Y_hat
    return np.mean(resid*resid)

def test(X_test, Y_test, w):
    X_test = np.r_[np.array([np.ones(X_test.shape[1])]), X_test]
    Y_hat = np.dot(w.T, X_test)
    error = mse(Y_hat, Y_test)
    return error

def k_cross_validation(K, lam, X_train_org, Y_train_org):
    CV_error = 0
    dk = Y_train_org.shape[0]//K
    for k in range(K):
        X_train = np.c_[X_train_org[:,:k],X_train_org[:,k+dk:]]
        X_valid = X_train_org[:,k:k+dk]
        Y_train = np.r_[Y_train_org[:k],Y_train_org[k+dk:]]
        Y_valid = Y_train_org[k:k+dk]
        X_train = sphere(X_train)[0]
        X_valid = sphere(X_valid)[0]
        w = train(X_train, Y_train, lam)[0]
        e = test(X_valid, Y_valid, w)
        CV_error += e
    CV_error /= K
    print("cv error with lambda = {:d} is {:.5f}".format(lam, CV_error,))
    return CV_error

X_train_org, Y_train_org, X_test, Y_test = load_data("boston-corrected")
X_train_valid, X_train_mu, X_train_sigma2 = sphere(X_train_org)
Y_train_valid = Y_train_org
print("X_mu vector is ", X_train_mu)
print("X_sigma2 vector is ", X_train_sigma2)

dfs = []
errors = []
for lam in range(21):
    w, df, error = train(X_train_valid, Y_train_valid, lam)
    dfs.append(df)
    errors.append(error)
plt.figure()
plt.xlabel("lambda")
plt.ylabel("effective degree of freedom")
plt.plot(range(21), dfs)
plt.show()

plt.figure()
plt.xlabel("lambda")
plt.ylabel("mean squared error")
plt.plot(range(21), errors)
plt.show()

w = train(X_train_valid, Y_train_valid, 0)[0]
X_test = sphere(X_test)[0]
e = test(X_test, Y_test, w)
print("mse in test set: {:.5f}".format(e))

lam_star = 0
df_star = 0
cve_min = float('inf')
for lam in range(201):
    cve = k_cross_validation(10, lam, X_train_org, Y_train_org)
    if cve < cve_min:
        cve_min = cve
        lam_star = lam
        df_star = train(X_train_valid, Y_train_valid, lam)[1]
print("lambda = {:d}, df = {:.3f}, CV error = {:.3f}".format(lam_star, df_star, cve_min))