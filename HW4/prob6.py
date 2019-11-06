import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()

# You have two features and two classifications
data_0, data_1 = iris.data[:,1:3][:50], iris.data[:,1:3][50:100]

# TODO: Find a LDA Boundary
mu_0, mu_1 = np.mean(data_0, axis=0), np.mean(data_1, axis=0)
sigma2_0 = np.cov(data_0.T, ddof = 0)
sigma2_1 = np.cov(data_1.T, ddof = 0)
sigma2 = (sigma2_0 + sigma2_1)/2
inv_sigma2 = np.linalg.pinv(sigma2)
w = np.dot((mu_1 - mu_0).T, inv_sigma2)
b = -(np.dot(np.dot(mu_1.T, inv_sigma2), mu_1) - np.dot(np.dot(mu_0.T, inv_sigma2), mu_1))/2

print("means in X0 is", mu_0)
print("means in X1 is", mu_1)
print("Corvariance is ")
print(sigma2)

x1min, x1max = min(np.min(data_0[:,0]), np.min(data_1[:,0])) - 0.5, max(np.max(data_0[:,0]), np.max(data_1[:,0])) + 0.5
x_1 = np.linspace(x1min, x1max, 100)
x_2 = (-b - w[0]*x_1)/w[1]
plt.plot(x_1, x_2)
plt.scatter(data_1[:,0], data_1[:,1], c = 'r')
plt.scatter(data_0[:,0], data_0[:,1], c = 'b')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("LDA boundary and scatter")
plt.show()

# TODO: Find a QDA Boundary
inv_sigma2_0 = np.linalg.pinv(sigma2_0)
inv_sigma2_1 = np.linalg.pinv(sigma2_1)
A = inv_sigma2_0 - inv_sigma2_1
b = np.dot(inv_sigma2_0, mu_0) - np.dot(inv_sigma2_1, mu_1)
c = 0.5*(np.log(np.linalg.det(sigma2_0)/np.linalg.det(sigma2_1)) + np.dot(mu_0.T, np.dot(inv_sigma2_0, mu_0)) - np.dot(mu_1.T, np.dot(inv_sigma2_1, mu_1)))

print("means in X0 is", mu_0)
print("means in X1 is", mu_1)
print("Corvariances in X0 is")
print(sigma2_0)
print("Corvariances in X1 is")
print(sigma2_1)

x1min, x1max = min(np.min(data_0[:,0]), np.min(data_1[:,0])) - 1, max(np.max(data_0[:,0]), np.max(data_1[:,0])) + 1
x2min, x2max = min(np.min(data_0[:,1]), np.min(data_1[:,1])) - 1, max(np.max(data_0[:,1]), np.max(data_1[:,1])) + 1
x_1 = np.linspace(x1min, x1max,100)
x_2 = np.linspace(x2min, x2max,100)
boundary = []
for i in x_2:
    for j in x_1:
        x = np.array([j,i])
        boundary.append(0.5*np.dot(np.dot(x.T, A), x) - np.dot(x.T, b) + c)
boundary = np.array(boundary).reshape(x_1.shape[0], x_2.shape[0])
x_1, x_2 = np.meshgrid(x_1, x_2)
plt.scatter(data_1[:,0], data_1[:,1], c = 'r')
plt.scatter(data_0[:,0], data_0[:,1], c = 'b')
plt.contour(x_1, x_2, boundary, levels = [0])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("QDA boundary and scatter")
plt.show()

# TODO: Make two scatterplots of the data, one showing the LDA Boundary and one showing the QDA Boundary