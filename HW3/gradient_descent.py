import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt

class linear_regression():
    def __init__(self, X, Y, learning_rate = 0.05):
        self.lr = learning_rate
        self.p, self.n = X.shape
        self.X, self.Y = sphere(X, Y)
        self.X = np.r_[np.array([np.ones(self.n)]), self.X]
        self.Y = np.array([self.Y])
        self.w = np.array([np.zeros(self.p+1)]).T

    def gradient_descent(self):
        grad = (2/self.n) * np.dot(self.X, (np.dot(self.w.T, self.X) - self.Y).T)
        self.w = self.w - self.lr*grad

    def stochastic_GD(self):
        X = self.X
        Y = self.Y
        for i in range(self.n):
            grad = np.array([X[:,i]*(np.dot(self.w.T, X[:,i]) - Y[:,i])]).T
            self.w = self.w - self.lr*grad

    def matrix_method(self):
        X = self.X
        Y = self.Y
        self.w = np.dot(np.dot(np.linalg.pinv(np.dot(X, X.T)), X), Y.T)

    def fit(self, X_test):
        X_test = np.r_[np.array([np.ones(X_test.shape[1])]), X_test]
        return np.dot(self.w.T, X_test)

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

def sphere(X, Y):
    n = Y.shape[0]
    X_mu = np.mean(X, 1)
    X_sigma2 = (np.mean(X*X, 1) - (X_mu*X_mu))*(n/(n-1))
    X = (X - X_mu[:, None]) / np.sqrt(X_sigma2[:,None])
    return X, Y

def mse(Y_hat, Y):
    resid = Y - Y_hat
    return np.mean(resid*resid)

def train(model, method = "BGD", K = 500):
    if method == "BGD":
        for _ in range(K):
            model.gradient_descent()
    elif method == "SGD":
        for _ in range(K):
            model.stochastic_GD()
    elif method == "MAT":
        model.matrix_method()
    return model

def model_display(filename, method = "BGD", lr = 0.05, K = 500):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    model = linear_regression(X_train, Y_train, lr)
    model = train(model, method, K)
    if method == "BGD":
        method_name = "Batch Gradient Descent"
    elif method == "SGD":
        method_name = "Stochastic Gradient Descent"
    elif method == "MAT":
        method_name = "Directly Computation"
    X_test, Y_test = sphere(X_test, Y_test)
    Y_hat = model.fit(X_test)
    error = mse(Y_hat, Y_test)
    #print(model.w.T)
    print("MSE on test set based on {} is {:3f}".format(method_name, error))

model_display("boston-corrected", method = "BGD", lr = 0.05, K = 500)
model_display("boston-corrected", method = "SGD", lr = 0.0005, K = 500)
model_display("boston-corrected", method = "MAT")