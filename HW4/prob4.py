import numpy as np 
import matplotlib.pyplot as plt

#(a)
def multivariate_gaussian(X, mu, sigma2):
    d = 1 if isinstance(X, float) or isinstance(X, int) else X.shape[0]
    coef = 1 / np.power(np.linalg.det(sigma2), 0.5) / np.power(2*np.pi, d/2)
    e = np.exp(-0.5*np.dot(np.dot((X - mu).T, np.linalg.pinv(sigma2)), (X-mu)))
    return coef*e

#(b)
def marginal_distribution(mu, sigma2, indeces):
    mu_U = mu[indeces]
    sigma2_U = sigma2[indeces][:,indeces].reshape(-1,len(indeces))
    return mu_U, sigma2_U

#(d)
def conditional_distribution(X, mu, sigma2, indeces):
    mu_U, sigma2_U = marginal_distribution(mu, sigma2, indeces)
    indeces_V = list(set(range(mu.shape[0])).difference(set(indeces)))
    mu_V, sigma2_V = marginal_distribution(mu, sigma2, indeces_V)
    sigma2_U_V = sigma2[indeces][:,indeces_V]
    sigma2_V_U = sigma2_U_V.T
    U = X[indeces]
    mu_V_U = mu_V + np.dot(np.dot(sigma2_V_U, np.linalg.pinv(sigma2_U)), U-mu_U)
    sigma2_V_U = sigma2_V - np.dot(np.dot(sigma2_V_U,np.linalg.pinv(sigma2_U)), sigma2_U_V)
    return mu_V_U, sigma2_V_U

#(c)
mu = np.zeros(2)
sigma2 = np.array([[1,0.5],[0.5,1]])
density = []
X_1_range = np.arange(-5,5,0.1)
idx = [0]
mu_1, sigma2_1 = marginal_distribution(mu, sigma2, idx)
for X_1 in X_1_range:
    p1 = multivariate_gaussian(X_1, mu_1, sigma2_1)
    density.append(p1)

plt.plot(X_1_range, density)
plt.xlabel("X1")
plt.ylabel("f(X1)")
plt.show()

#(e)
mu = np.array([0.5,0,-0.5,0])
sigma2 = np.array([[1,0.5,0,0],[0.5,1,0,1.5],[0,0,2,0],[0,1.5,0,4]])
density = []
X_1_range = np.arange(-3,3,0.1) + mu[0]
X_4_range = np.arange(-3,3,0.1) + mu[3]
X = np.array([0,0.1,-0.2,0])
idx = [1,2]
mu_14_23, sigma2_14_23 = conditional_distribution(X, mu, sigma2, idx)
print("mu = ", mu_14_23)
print("Sigma2 = ", sigma2_14_23)
for X_4 in X_4_range:
    for X_1 in X_1_range:
        X[0] = X_1
        X[3] = X_4
        X_14 = X[[0,3]]
        p = multivariate_gaussian(X_14, mu_14_23, sigma2_14_23)
        density.append(p)
density = np.array(density).reshape(len(X_1_range), len(X_4_range))
X_1_range, X_4_range = np.meshgrid(X_1_range, X_4_range)

plt.contourf(X_1_range, X_4_range, density, 8, cmap = 'jet')
cntr = plt.contour(X_1_range, X_4_range, density, 8, colors='black',linewidths=0.5)
plt.clabel(cntr, inline_spacing=1, fmt='%.2f', fontsize=8)
plt.xlabel("X1")
plt.ylabel("X4")
plt.show()