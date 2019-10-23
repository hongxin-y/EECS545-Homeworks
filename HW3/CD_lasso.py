import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
np.random.seed(0)

class CD_linear_regression():
    def __init__(self, X, Y, lam = 0.001):
        self.lam = lam
        self.p, self.n = X.shape
        self.X = np.r_[np.ones(self.n)[None,:], X]
        self.Y = Y
        self.w = np.zeros(self.p+1)[:,None]

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
                
                if ci > self.lam:
                    self.w[i,0] = (ci - self.lam)/(ai)
                elif ci < -self.lam:
                    self.w[i,0] = (ci + self.lam)/(ai)
                else:
                    self.w[i,0] = 0

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

def train(model, method = "CD", K = 500, plot = False):
    evaluations = []
    errors = []
    if method == "CD":
        errors, evaluations = model.coodinate_descent(K)
    evaluations = np.array(evaluations).T[0]

    if plot:
        plt.figure()
        plt.xlabel("Iteration k")
        plt.ylabel("coefficients")
        #print(evaluations.shape)
        for i in range(1, evaluations.shape[0]):
            plt.plot(range(1, evaluations[i].shape[0]+1), evaluations[i], label= "w"+str(i))
            #plt.annotate("w%d "%i, xy = (K, evaluations[i,-1]))
        
        plt.legend()
        plt.show()
        plt.figure()
        plt.xlabel("Iteration k")
        plt.ylabel("mse on training dataset")
        plt.plot(range(1,len(errors)+1), errors)
        plt.show()
    return model

def test(X_test, Y_test, w):
    X_test = np.r_[np.ones(X_test.shape[1])[None,:], X_test]
    Y_hat = np.dot(w.T, X_test)
    error = mse(Y_hat, Y_test)
    return error

def k_cross_validation(K, lam, X_train_valid, Y_train_valid, N = 300):
    CV_error = 0
    dk = Y_train_valid.shape[0]//K
    for i in range(K):
        k = dk*i
        X_train = np.c_[X_train_valid[:,:k],X_train_valid[:,k+dk:]]
        X_valid = X_train_valid[:,k:k+dk]
        Y_train = np.r_[Y_train_valid[:k],Y_train_valid[k+dk:]]
        Y_valid = Y_train_valid[k:k+dk]
        model = CD_linear_regression(X_train, Y_train, lam)
        model = train(model, K = N, plot = False)
        e = test(X_valid, Y_valid, model.w)
        CV_error += e
    CV_error /= K
    print("cv error with lambda = {:.3f} is {:.5f}".format(lam, CV_error))
    return CV_error

def model_display(filename, method = "CD", K = 500, lam = 0.001):
    X_train_valid, Y_train_valid, X_test, Y_test = load_data(filename)
    X_train_valid, X_mu, X_sigma2 = sphere(X_train_valid)
    X_test = (X_test - X_mu[:, None]) / np.sqrt(X_sigma2[:, None])
    
    model = CD_linear_regression(X_train_valid, Y_train_valid, lam)
    model = train(model, method, K, plot = True)
    error = test(X_test, Y_test, model.w)
    print("mse on test dataset with lambda = {:.2f} is {:.3f}".format(lam, error))
    print("The weights are")
    print(model.w.T)
    
    cvs = []
    min_cv = float("inf")
    lam_star = 0
    for i in range(0,1001):
        lam = i/10
        cv = k_cross_validation(10, lam, X_train_valid, Y_train_valid, K)
        cvs.append(cv)
        if cv < min_cv:
            min_cv = cv
            lam_star = lam

    plt.figure()
    plt.xlabel("lambda")
    plt.ylabel("cross validation error")
    plt.plot(np.array(range(0,1001))/10, cvs)
    plt.show()
    print("The best lambda is {:.3f} with cross validation error {:.3f}.".format(lam_star, min_cv))
    model_best = CD_linear_regression(X_train_valid, Y_train_valid, lam_star)
    model_best = train(model_best, method, K, plot = True)
    error = test(X_test, Y_test, model_best.w)
    print("mse on test dataset is {:.3f}".format(error))

    print("weights are")
    print(model_best.w.T)
    
    return model_best

model_display("boston-corrected", method = "CD", K = 700, lam = 100)