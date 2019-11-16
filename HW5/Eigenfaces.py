import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

yale = sio.loadmat('yalefaces.mat')
yalefaces = yale['yalefaces']
r, c, n = yalefaces.shape
X = yalefaces.reshape(r*c, n)
cov = np.cov(X)
eigenvalues, U = np.linalg.eig(cov)
idx = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[idx]
U = U[:,idx]
plt.semilogy(eigenvalues)
plt.show()

total = np.dot(eigenvalues, np.ones(eigenvalues.shape))
percentage = np.cumsum(eigenvalues) / total
idx_95 = np.where(percentage>=0.95)[0][0]
idx_99 = np.where(percentage>=0.99)[0][0]

print("95% data can be explained within {} components.".format(idx_95))
print("99% data can be explained within {} components.".format(idx_99))

fig, ax = plt.subplots(nrows=4, ncols=5)
x = np.mean(X, axis=1).reshape(r,c)
ax[0][0].imshow(x, extent=[0, 1, 0, 1], cmap=plt.get_cmap('gray'))
for i in range(1,20):
    x = U[:,i-1].reshape(r,c)
    ax[i//5][i%5].imshow(x, extent=[0, 1, 0, 1], cmap=plt.get_cmap('gray'))
plt.show()
