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
    print(Y.shape)
    X_train = X[:,:-46]
    Y_train = Y[:-46]
    X_test = X[:,-46:]
    Y_test = Y[-46:]
    return X_train, Y_train, X_test, Y_test

def sphere(X, Y):
    n = Y.shape[0]
    X_mu = np.mean(X, 1)
    Y_mu = np.mean(Y)
    X_sigma2 = (np.mean(X*X, 1) - (X_mu*X_mu))*(n/(n-1))
    Y_sigma2 = np.sum((Y - Y_mu)*(Y- Y_mu))/(n-1)
    X = (X - X_mu[:, None]) / X_sigma2[:,None]
    Y = (Y - Y_mu)/Y_sigma2
    return X, Y, X_mu, Y_mu, X_sigma2, Y_sigma2

def train(X_train, Y_train, lam):
    X_train = np.r_[np.array([np.ones(X_train.shape[1])]), X_train]
    p, n = X_train.shape[0] - 1, X_train.shape[1]
    I_hat = np.eye(p+1)
    I_hat[0][0] = 0
    A = np.dot(X_train, X_train.T) + lam*I_hat
    L = np.dot(np.linalg.inv(A), X_train)
    w = np.dot(L, Y_train)
    df = np.trace(np.dot(X_train.T, L))

    Y_hat = np.dot(X_train.T, w)
    #print(Y_hat)
    error = mse(Y_hat, Y_train)
    return w, df, error

def mse(Y_hat, Y_train):
    resid = Y_train - Y_hat
    return np.mean(resid*resid)

def test(X_test, Y_test, w):
    X_test = np.r_[np.array([np.ones(X_test.shape[1])]), X_test]
    Y_hat = np.dot(X_test.T, w)
    error = mse(Y_hat, Y_test)
    return error

def k_cross_validation(K, lam, X_train_org, Y_train_org):
    CV_error = 0
    dk = Y_train_org.shape[0]//K
    print(dk)
    for k in range(K):
        X_train = np.c_[X_train_org[:,:k],X_train_org[:,k+dk:]]
        X_valid = X_train_org[:,k:k+dk]
        Y_train = np.r_[Y_train_org[:k],Y_train_org[k+dk:]]
        Y_valid = Y_train_org[k:k+dk]
        X_train, Y_train = sphere(X_train, Y_train)[:2]
        X_valid, Y_valid = sphere(X_valid, Y_valid)[:2]
        w = train(X_train, Y_train, lam)[0]
        e = test(X_valid, Y_valid, w)
        CV_error += e
    print("cv error with lam =", lam, "is", CV_error/K)
    return CV_error/K

X_train_org, Y_train_org, X_test, Y_test = load_data("boston-corrected")
X_train_valid, Y_train_valid, X_train_mu, Y__train_mu, X_train_sigma2, Y_train_sigma2 = sphere(X_train_org, Y_train_org)
#print(X,Y)

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
X_test, Y_test = sphere(X_test, Y_test)[0:2]
e = test(X_test, Y_test, w)
print("mse in test set:", e)

lam_star = 0
df_star = 0
cve_min = float('inf')
for lam in range(21):
    cve = k_cross_validation(10, lam, X_train_org, Y_train_org)
    if cve < cve_min:
        cve_min = cve
        lam_star = lam
        df_star = train(X_train_valid, Y_train_valid, lam)[1]
print(lam_star, df_star, cve_min)
