import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
np.random.seed(0)

class CD_linear_regression():
    def __init__(self, X, Y, lam = 0.001):
        self.lam = lam
        self.p, self.n = X.shape
        self.X = np.r_[np.array([np.ones(self.n)]), X]
        self.Y = Y
        self.w = np.array([np.zeros(self.p+1)]).T

    def coodinate_descent(self, K):
        errors = []
        evaluations = []
        it = 0
        while it < K:
            self.w[0,0] = np.mean(self.Y)
            Y_hat = np.dot(self.w.T, self.X)
            error = mse(Y_hat, self.Y)
            errors.append(error)
            evaluations.append(np.copy(self.w))
            for i in range(1, self.p+1):
                ai = 2*np.dot(self.X[i], self.X[i].T)
                y_hat = np.dot(self.w.T, self.X) - self.w[i,0]*self.X[i,:]
                ci = 2*np.dot(self.Y - y_hat, self.X[i,:])
                self.w[i,0] = ci/(ai+2*self.lam)

                Y_hat = np.dot(self.w.T, self.X)
                error = mse(Y_hat, self.Y)
                errors.append(error)
                evaluations.append(np.copy(self.w))
                it += 1
        return errors, evaluations

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

def mse(Y_hat, Y):
    resid = Y - Y_hat
    return np.mean(resid*resid)

def train(model, method = "CD", K = 500):
    evaluations = []
    errors = []
    if method == "CD":
        errors, evaluations = model.coodinate_descent(K)
    evaluations = np.array(evaluations).T[0]
    plt.figure()
    plt.xlabel("number of iterations")
    plt.ylabel("coefficients except w0")
    #print(evaluations.shape)
    for i in range(1, evaluations.shape[0]):
        plt.plot(range(1, evaluations[i].shape[0]+1), evaluations[i], label= "w"+str(i))
    plt.legend()
    plt.show()
    plt.figure()
    plt.xlabel("number of iterations")
    plt.ylabel("mse on training dataset")
    plt.plot(range(1,len(errors)+1), errors)
    plt.show()
    return model

def test(X_test, Y_test, w):
    X_test = np.r_[np.array([np.ones(X_test.shape[1])]), X_test]
    Y_hat = np.dot(w.T, X_test)
    error = mse(Y_hat, Y_test)
    return error

def model_display(filename, method = "CD", K = 500, lam = 0.001):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    X_train, X_mu, X_sigma2 = sphere(X_train)
    X_test = (X_test - X_mu[:, None])/ np.sqrt(X_sigma2[:, None])
    model = CD_linear_regression(X_train, Y_train, lam)
    model = train(model, method, K)
    if method == "CD":
        method_name = "Coodinate Descent"
    error = test(X_test, Y_test, model.w)
    print("mse on test dataset is {:.3f}".format(error))
    print("The weights are")
    print(model.w.T)
    return model

model_display("boston-corrected", method = "CD", K = 700, lam = 100)