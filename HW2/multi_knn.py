import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class multi_knn_classifier():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.c_map = {}
        self.one_hot_encoder()

    def one_hot_encoder(self):
        n = len(set(list(self.Y)))
        new_Y = np.array([])
        idx = 0
        c_map = {}
        for y in self.Y:
            if y not in c_map:
                c_map[y] = idx
                idx += 1
        new_Y = [[ 1 if c_map[y] == i else 0 for i in range(n)] for y in self.Y]
        self.c_map = c_map
        self.Y = np.array(new_Y)

    def fit(self, k, x):
        X = np.array(self.X)
        Y = np.array(self.Y)
        x = np.array(x)
        n = X.shape[0]
        c = Y.shape[1]
        new_X = np.r_[X,[x]]
        Y = np.r_[Y, [np.zeros(c)]]
        G = np.dot(new_X, np.transpose(new_X))
        d = np.array([np.diag(G)])
        one = np.array([np.ones(n+1)])
        D = np.dot(np.transpose(one), d) - 2*G + np.dot(np.transpose(d),one)
        idx = np.argsort(D[:,-1], 0)[0:k+1]
        Y_vote = Y[idx]
        Y_majority = np.sum(Y_vote, 0)
        return np.argmax(Y_majority)
        
def train(train_filename):
    df = pd.read_csv("iris.train.csv")
    print(df)
    X_train = np.array(df)[:,1:-1]
    Y_train = np.array(df)[:,-1]
    model = multi_knn_classifier(X_train, Y_train)
    return model

def test(k, classifier, test_filename):
    df = pd.read_csv(test_filename)
    X_Y = np.array(df)
    acc = 0
    for x in X_Y:
        y = x[-1]
        x = x[1:-1]
        #print(x.shape)
        c = classifier.fit(k, x)
        if c == classifier.c_map[y]:
            acc += 1
    acc /= len(X_Y)
    print(acc)
    return acc

model = train("iris.train.csv")
K = range(1,51)
accs = []
for k in K:
    acc = test(k, model, "iris.test.csv")
    accs.append(acc)
plt.figure()
plt.xlabel("k")
plt.ylabel("accuracy in test set")
plt.plot(K, accs)
plt.show()