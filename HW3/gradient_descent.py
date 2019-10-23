import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt

class linear_regression():
    def __init__(self, X, Y, learning_rate = 0.05):
        self.lr = learning_rate
        self.p, self.n = X.shape
        self.X = np.r_[np.array([np.ones(self.n)]), X]
        self.Y = np.array([Y])
        self.w = np.array([np.zeros(self.p+1)]).T

    def gradient_descent(self, K):
        train_errors = []
        for k in range(K):
            grad = (2/self.n) * np.dot(self.X, (np.dot(self.w.T, self.X) - self.Y).T)
            self.w = self.w - self.lr*grad
            Y_hat = np.dot(self.w.T, self.X)
            train_errors.append(mse(Y_hat, self.Y))
        return train_errors

    def stochastic_GD(self, K):
        train_errors = []
        for k in range(K):
            idx = np.random.permutation(range(self.n))
            X = self.X[:,idx]
            Y = self.Y[:,idx]
            for i in range(self.n):
                grad = 2 * np.array([X[:,i]*(np.dot(self.w.T, X[:,i]) - Y[:,i])]).T
                self.w = self.w - self.lr*grad
            Y_hat = np.dot(self.w.T, self.X)
            train_errors.append(mse(Y_hat, self.Y))
        return train_errors

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

def sphere(X):
    X_mu = np.mean(X, 1)
    X_std = np.std(X, 1, ddof = 1)
    X_sigma2 = X_std * X_std
    X = (X - X_mu[:, None]) / X_std[:,None]
    return X, X_mu, X_sigma2

def mse(Y_hat, Y):
    resid = Y - Y_hat
    return np.mean(resid*resid)

def train(model, method = "BGD", K = 500):
    train_errors = []
    if method == "BGD":
        train_errors = model.gradient_descent(K)
    elif method == "SGD":
        train_errors = model.stochastic_GD(K)
    elif method == "MAT":
        model.matrix_method()
    if train_errors != []:
        plt.figure()
        plt.xlabel("Iteration k")
        plt.ylabel("trainning error")
        plt.plot(range(1,K+1), train_errors)
        plt.show()
    return model

def model_display(filename, method = "BGD", lr = 0.05, K = 500):
    X_train, Y_train, X_test, Y_test = load_data(filename)
    X_train, X_mu, X_sigma2 = sphere(X_train)
    model = linear_regression(X_train, Y_train, lr)
    model = train(model, method, K)
    if method == "BGD":
        method_name = "Batch Gradient Descent"
    elif method == "SGD":
        method_name = "Stochastic Gradient Descent"
    elif method == "MAT":
        method_name = "Directly Computation"
    Y_hat = model.fit(X_train)
    error = mse(Y_hat, Y_train)
    print("MSE on training set based on {} is {:3f}".format(method_name, error))

    X_test = (X_test - X_mu[:,None]) / np.sqrt(X_sigma2[:,None])
    Y_hat = model.fit(X_test)
    error = mse(Y_hat, Y_test)
    print("MSE on testing set based on {} is {:3f}".format(method_name, error))
    print("The learnd weight of {} is ".format(method_name))
    print(model.w.T)
    return model

model_display("boston-corrected", method = "BGD", lr = 0.05, K = 500)
model_display("boston-corrected", method = "SGD", lr = 0.0005, K = 500)
model_display("boston-corrected", method = "MAT")