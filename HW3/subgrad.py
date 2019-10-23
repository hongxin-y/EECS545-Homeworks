import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
np.random.seed(0)

class linear_regression():
    def __init__(self, X, Y, learning_rate = 0.05, lam = 0.001):
        self.lr = learning_rate
        self.lam = lam
        self.p, self.n = X.shape
        self.X = np.r_[np.array([np.ones(self.n)]), X]
        self.Y = Y
        self.w = np.array([np.zeros(self.p+1)]).T

    def obj_grad(self, X, Y):
        eps = 10**(-5)
        n = self.X.shape[1]
        w_1 = np.copy(self.w)
        w_1[0,0] = 0
        lam_w1 = 1/n * (self.lam*w_1).T
        Y_X_i = (Y[0,:] * X).T/n
        low = np.where(Y_X_i > 0, lam_w1 - Y_X_i, lam_w1)
        high = np.where(Y_X_i < 0, lam_w1 - Y_X_i, lam_w1)
        subgrad = np.random.uniform(low, high)
        g = np.zeros(Y_X_i.shape)
        Y_Y_hat = (Y*np.dot(self.w.T, X)).T
        g = np.where(Y_Y_hat > 1, lam_w1, lam_w1 - Y_X_i)
        g = np.where(np.abs(Y_Y_hat - 1) <= eps, subgrad, g)
        return np.array([np.sum(g, axis = 0)]).T

    def obj_func(self):
        A = 1 - self.Y*(np.dot(self.w.T, self.X))
        L = np.where(A > 0, A, 0)
        w1 = np.copy(self.w)
        w1[0,0] = 0
        J = np.sum(L) / self.n + self.lam / 2 * np.dot(w1.T, w1)[0,0]
        return J

    def gradient_descent(self, K):
        obj_vals = []
        for k in range(K):
            lr = (100/(k+1)) if self.lr == 'non-const' else self.lr
            grad = self.obj_grad(self.X, self.Y)
            self.w = self.w - lr*grad
            print("Iteration: {:d}, objective function: {:3f}".format(k+1,self.obj_func()))
            obj_vals.append(self.obj_func())
        return obj_vals

    def stochastic_GD(self, K):
        obj_vals = []
        for k in range(K):
            idx = np.random.permutation(range(self.n))
            X = self.X[:,idx]
            Y = self.Y[:,idx]
            lr = (100/(k+1)) if self.lr == 'non-const' else self.lr
            for i in range(self.n):
                grad = self.obj_grad(np.array([X[:,i]]).T, np.array([Y[:,i]]))
                #print(i, ": ", grad)
                self.w = self.w - lr*grad
            obj = self.obj_func()
            print("Iteration: {:d}, objective function: {:3f}".format(k+1,obj))
            obj_vals.append(obj)
        return obj_vals
        
    def fit(self, X_test):
        X_test = np.r_[np.array([np.ones(X_test.shape[1])]), X_test]
        return np.dot(self.w.T, X_test)

def load_data(filename):
    data = scio.loadmat(filename)
    X_train, Y_train = data['x'], data['y']
    return X_train, Y_train

def mse(Y_hat, Y):
    resid = Y - Y_hat
    return np.mean(resid*resid)

def train(model, method = "BGD", K = 500):
    obj_vals = []
    obj_vals = model.gradient_descent(K) if method == "BGD" else model.stochastic_GD(K)
    plt.figure()
    plt.xlabel("Iteration k")
    plt.ylabel("objective function")
    plt.plot(range(1,len(obj_vals)+1), obj_vals)
    plt.show()
    print("The learned weight if")
    print(model.w.T)
    print("The minimal achieved objective function is {:.3f}".format(min(obj_vals)))
    return model

def model_display(filename, method = "BGD", lr = 0.05, K = 500, lam = 0.001):
    X_train, Y_train = load_data(filename)
    model = linear_regression(X_train, Y_train, lr, lam)
    model = train(model, method, K)
    method_name = "Batch Gradient Descent" if method == "BGD" else "Stochastic Gradient Descent"
    plt.figure()
    plt.xlabel('x1')
    plt.ylabel('x2')
    p_idx = np.where(Y_train[0,:] == 1)[0]
    n_idx = np.where(Y_train[0,:] == -1)[0]
    plt.scatter(X_train[0,p_idx], X_train[1,p_idx], marker = '.', color = 'c')
    plt.scatter(X_train[0,n_idx], X_train[1,n_idx], marker = '.', color = 'r')
    x1 = np.linspace(0, 8, 100)
    x2 = x1*(-model.w[1][0]/model.w[2][0]) + (-model.w[0][0]/model.w[2][0])
    plt.plot(x1, x2)
    plt.show()
    return model

model_display("nuclear.mat", method = "BGD", lr = "non-const", K = 50, lam = 0.001)
model_display("nuclear.mat", method = "SGD", lr = "non-const", K = 50, lam = 0.001)