import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)
n = 200 #sample size
K = 2 #number of lines
e = np.array([0.7,0.3]) #mixing weights
w = np.array([-2,1]) #slopes of lines
b = np.array([0.5,-0.5]) #offsets of lines
v = np.array([0.2,0.1]) #variances
x = np.zeros([n])
y = np.zeros([n])
for i in range(0,n):
    x[i] = np.random.rand(1)
    if np.random.rand(1) < e[0]:
        y[i] = w[0]*x[i] + b[0] + np.random.randn(1)*np.sqrt(v[0])
    else:
        y[i] = w[1]*x[i] + b[1] + np.random.randn(1)*np.sqrt(v[1])

plt.figure()
plt.ion()

# initialization
eps_hat = np.array([.5 , .5])
weights_hat = np.array([1. ,-1.])
bias_hat = np.array([0., 0.])
var_hat = np.array([np.var(y),np.var(y)])
eps_kj = np.ones((K, n))/2
Q_old = float("-inf")
x_extend = np.r_[np.ones((1,x.shape[0])), x[None, :]]
Qs = []
t = 1
while True:
    # E-step
    for k in range(K):
        eps_kj[k] = norm.pdf(y, weights_hat[k] * x + bias_hat[k], np.sqrt(var_hat[k])) * eps_hat[k]
    eps_kj /= np.sum(eps_kj, axis=0)


    # M-step
    for k in range(K):
        eps_hat[k] = np.mean(eps_kj[k])
        var_hat[k] = np.sum(eps_kj[k] * (y - np.dot(weights_hat[k], x) - bias_hat[k]) * (y - np.dot(weights_hat[k], x) - bias_hat[k])) / np.sum(eps_kj[k])
        C = np.diag(eps_kj[k])
        M = np.dot(x_extend, C)
        A = np.dot(M, x_extend.T)
        W = np.dot(np.dot(np.linalg.inv(A), M), y)
        bias_hat[k] = W[0]
        weights_hat[k] = W[1:]

    # log-likelihood
    sum_density = np.zeros(K)
    for k in range(K):
        sum_density[k] = np.dot(np.log(eps_hat[k] * norm.pdf(y, weights_hat[k] * x + bias_hat[k], np.sqrt(var_hat[k]))), eps_kj[k])
    Q_new = np.sum(sum_density)
    Qs.append(Q_new)

    plt.cla()
    plt.plot(x,y,'bo')
    scatter = np.linspace(0, 1, num=100)
    plt.plot(scatter,w[0]*scatter+b[0],'k')
    plt.plot(scatter,w[1]*scatter+b[1],'k')
    plt.plot(scatter,weights_hat[0]*scatter+bias_hat[0],'.')
    plt.plot(scatter,weights_hat[1]*scatter+bias_hat[1],'.')
    plt.title("Iteration " + str(t))
    plt.pause(0.1)
    # plt.imshow()
    

    # stoping criterion
    if Q_new - Q_old < 1e-4:
        break
    Q_old = Q_new
    t += 1

print("The iteration number is {}.".format(t))
print("Estimated Parameters: ")
print("Weight: ", weights_hat)
print("Bias: ", bias_hat)
print("Variance: ", var_hat)
print("Epsilon: ", eps_hat)

# plot
plt.ioff()
plt.show()

plt.plot(range(1, 1 + len(Qs)), Qs)
plt.xlabel('Iteration number')
plt.ylabel('log likelihood')
plt.show()

plt.plot(x,y,'bo')
t = np.linspace(0, 1, num=100)
plt.plot(t,w[0]*t+b[0],'k')
plt.plot(t,w[1]*t+b[1],'k')
plt.plot(t,weights_hat[0]*t+bias_hat[0],'.')
plt.plot(t,weights_hat[1]*t+bias_hat[1],'.')
plt.show()
