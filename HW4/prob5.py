import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def marginal_distribution(mu, sigma2, indeces):
    mu_U = mu[indeces]
    sigma2_U = sigma2[indeces][:,indeces].reshape(-1,len(indeces))
    return mu_U, sigma2_U

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

def display_prior_Gaussian_Process(mu, sigma, lim):
    np.random.seed(0)
    xmin, xmax = lim
    X = np.arange(xmin, xmax,0.1)
    cov = np.array([[np.exp(-(X[i] - X[j])*(X[i] - X[j])/(2*sigma*sigma))  for i in range(X.shape[0])] for j in range(X.shape[0])])
    Y = np.random.multivariate_normal(np.array([mu]*X.shape[0]), cov, 3)
    plt.figure()
    plt.plot(X, Y.T)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def display_posterior_Gaissian_Process(X_post, Y_post, X_pred, sigma):
    np.random.seed(0)
    xmin, xmax = min(X_pred), max(X_pred)
    X = np.r_[X_post, X_pred]
    Y = np.r_[Y_post, np.zeros(X_pred.shape[0])]
    mu = np.zeros(X.shape[0])
    cov = np.array([[np.exp(-(X[i] - X[j])*(X[i] - X[j])/(2*sigma*sigma))  for i in range(X.shape[0])] for j in range(X.shape[0])])
    mu_pred, sigma2_pred = conditional_distribution(Y, mu, cov, list(range(X_post.shape[0])))
    
    line = plt.plot(X_pred, mu_pred)
    plt.setp(line, linewidth = 3.0)
    for i in range(5):
        Y_pred = np.random.multivariate_normal(mu_pred, sigma2_pred)
        line = plt.plot(X_pred, Y_pred)
        plt.setp(line, linewidth = 1.0)
    plt.scatter(X_post, Y_post, marker='x', zorder=100, color = 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return 

sigma_list = [0.3, 0.5, 1.0]

#(a)
for sigma in sigma_list:
    display_prior_Gaussian_Process(0, sigma, (-5,5))

#(c)
X_S = np.array([-1, 2.4, -2.5, -3.3, 0.3])
Y_S = np.array([2, 5.2, -1.5, -0.8, 0.3])
xmin, xmax, step = -4, 3, 0.05
for sigma in sigma_list:
    display_posterior_Gaissian_Process(X_S, Y_S, np.arange(xmin, xmax, step), sigma)
