import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

class logistic_regression():
    #initialize parameter w with zeros vector and calculate corresponding other values including X^Tw, sigmoid(X^Tw), gradient and hessian
    #here X is of (d+1),n dimensions, Y is of 1,n dimensions and w is of (d+1),1 dimensions
    def __init__(self, X, Y):
        self.lam = 10
        self.d, self.n = X.shape
        self.w = np.transpose(np.array([np.zeros(self.d+1)]))
        self.X = np.r_[[np.ones(self.n)],X]
        self.Y = Y
        self.deriv()

    #this function is to calculate the generated values such as gradient and hessian matrix with current parameter w.
    #P is the vector of entries y_j*P(y=y_j) where j from 1 to n, and of n,1 dimensions
    def deriv(self):
        self.linear_part = np.dot(np.transpose(self.X), self.w)
        self.P = 1 / (np.exp(-np.transpose(self.Y)*self.linear_part) + 1)
        grad = -np.dot(self.X, (1-self.P)*np.transpose(self.Y)) + 2*self.lam*self.w
        A = np.diag(np.transpose(self.P*(1-self.P))[0])
        hess = np.dot(np.dot(self.X, A), np.transpose(self.X)) + 2*self.lam*np.eye(self.d+1)
        self.hessian = hess
        self.grad = grad

    #this function is to update the w using newton method.
    def newton_method(self, iterations):
        for i in range(iterations):
            if(np.linalg.det(self.hessian) == 0):
                print("early termination: %d"%iterations)
                break
            self.w = self.w - np.dot(np.linalg.inv(self.hessian), self.grad)
            self.deriv()
            val = -np.sum(np.log(self.P)) + self.lam*np.sqrt(np.dot(np.transpose(self.w), self.w))
            print("objective function: %.2f"%val)

    #this function is to do the calssification work using given X, the threold is 0.5.
    def forward(self, X):
        X = np.r_[[np.ones(X.shape[1])],X]
        linear_predict = np.dot(np.transpose(X), self.w)
        sigmoid_predict = 1 / (np.exp(-linear_predict) + 1) - 0.5
        return sigmoid_predict

#load data from root directory
def load_data(filename):
    mnist_49_3000 = sio.loadmat(filename)
    X = mnist_49_3000['x']
    Y = mnist_49_3000['y']
    d,n = X.shape
    X_train, Y_train = X[:,:2000], Y[:,:2000]
    X_test, Y_test = X[:,2000:], Y[:,2000:]
    return X_train, Y_train, X_test, Y_test

#train the logistic regression model and return the model.
def train(X, Y):
    model = logistic_regression(X, Y)
    model.newton_method(100)
    return model

#test the logistic regression model and retuen the accuracy on the test set.
def test(model, X, Y):
    results = model.forward(X)
    acc = 100*np.sum(np.sign(results) == np.transpose(Y))/(Y.shape[1])
    print("The accuracy on the test set is %.2f%%"%(acc))
    print("The error on the test set is %.2f%%"%(100-acc))
    conf = np.abs(np.transpose(results) - Y)
    #only consider the false cases, set the confidence of correct results as 0 and set the misclassified results as abs(sigmoid(X^Tx)-0.5)
    adjusted_conf = np.where(conf < 1, np.abs(np.transpose(results)), 0)
    idx = np.argsort(adjusted_conf, 1)
    idx = idx[:,:20][0]
    X_plt = X[:,idx]
    Y_plt = Y[:,idx]
    label_plt = np.transpose(results[idx])
    d = X.shape[0]
    plt.figure(figsize=(9,7))
    plt.subplots_adjust(wspace =0.5, hspace =0.5)
    for i in range(20):
        ax = plt.subplot(4,5,i+1)
        ax.set_title("True: %d, Labeled: %d"%(4 if Y_plt[0][i] < 0 else 9, 4 if label_plt[0][i] < 0 else 9),fontsize=10)
        plt.imshow( np.reshape(X_plt[:,i], (int(np.sqrt(d)), int(np.sqrt(d)))) )
    plt.show()
    return acc

X_train, Y_train, X_test, Y_test = load_data('mnist_49_3000.mat')
model = train(X_train, Y_train)
acc = test(model, X_test, Y_test)