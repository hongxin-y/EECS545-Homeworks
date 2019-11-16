import numpy as np

# load the training data 
train_features = np.load("spam_train_features.npy")
train_labels = np.load("spam_train_labels.npy")

# load the test data 
test_features = np.load("spam_test_features.npy")
test_labels = np.load("spam_test_labels.npy")

# prior Parameters
alpha = 1.0
beta = 1.0
n_1 = np.sum(train_labels)
n_0 = train_labels.shape[0] - n_1
n_i1 = np.dot(train_labels,train_features)
n_i0 = np.dot(1-train_labels, train_features)

# parameters estimation
pi = (n_1)/(n_0+n_1)
theta_1 = (n_i1 + alpha) / (n_1 + alpha + beta)
theta_0 = (n_i0 + alpha) / (n_0 + alpha + beta)

# calculate results
res = np.dot(test_features, np.log(theta_1 / theta_0)) + np.dot(1 - test_features, np.log((1-theta_1) / (1-theta_0))) - np.log((1-pi)/pi) > 0
err = np.sum(res!=test_labels)/test_labels.shape[0]

print("Accuracy on test data is {}".format(err))