import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
np.random.seed(0)

class layer():
    def __init__(self, dim_input, dim_output, activation = None, batch_first = True):
        self.batch_first = batch_first
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weights = np.random.randn(dim_input, dim_output)
        self.bias = np.random.randn(dim_output)
        self.activation = ReLU() if activation == 'relu' else None
        self.activation_derivative = None
        self.input = None
        self.batch_size = None

    def forward(self, x):
        self.input = x if self.batch_first == True else x.T
        self.batch_size = self.input.shape[0]
        linear = np.dot(self.weights.T, self.input.T) + np.dot(self.bias[:,None], np.ones((1, self.batch_size)))
        self.output = self.activation.forward(linear).T if self.activation != None else linear.T
        self.activation_derivative = self.activation.backward().T if self.activation != None else np.ones(self.output.shape)
        return self.output

class FCN():
    def __init__(self, input_dim, batch_first = True):
        self.hidden_layer1 = layer(input_dim, 64, activation="relu", batch_first = batch_first)
        self.hidden_layer2 = layer(64, 16, activation="relu", batch_first = batch_first)
        self.output_layer = layer(16, 1, activation=None, batch_first = batch_first)

        self.W1, self.b1 = self.hidden_layer1.weights, self.hidden_layer1.bias
        self.W2, self.b2 = self.hidden_layer2.weights, self.hidden_layer2.bias
        self.W3, self.b3 = self.output_layer.weights, self.output_layer.bias

        self.loss = None
        self.batch_size = 1
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.shape) == 1: x = x[None, :] if self.batch_first == True else x[:, None]
        output = self.hidden_layer1.forward(x)
        output = self.hidden_layer2.forward(output)
        output = self.output_layer.forward(output)
        self.output = output
        self.batch_size = x.shape[0]
        return output

    def backward(self, ytrue):
        self.loss = MSloss(ytrue)
        batch_loss = self.loss.forward(self.output)/self.batch_size

        self.dL_dyhat = self.loss.backward()/self.batch_size
        dL_dz = (self.dL_dyhat * self.output_layer.activation_derivative)
        self.dL_dW3 = np.dot(self.output_layer.input.T, dL_dz)
        self.dL_db3 = np.dot(np.ones(self.batch_size), dL_dz)
        self.dL_dh2 = np.dot(dL_dz, self.output_layer.weights.T)

        dL_dz = (self.dL_dh2 * self.hidden_layer2.activation_derivative)
        self.dL_dW2 = np.dot(self.hidden_layer2.input.T, dL_dz)
        self.dL_db2 = np.dot(np.ones(self.batch_size), dL_dz)
        self.dL_dh1 = np.dot(dL_dz, self.hidden_layer2.weights.T)

        dL_dz = (self.dL_dh1 * self.hidden_layer1.activation_derivative)
        self.dL_dW1 = np.dot(self.hidden_layer1.input.T, dL_dz)
        self.dL_db1 = np.dot(np.ones(self.batch_size), dL_dz)
        # self.dL_dh1 = np.dot(self.hidden_layer1.weights.T, dL_dz)
        return batch_loss

    def update(self, lr = 1e-7):
        self.b1 -= lr*self.dL_db1
        self.W1 -= lr*self.dL_dW1
        self.b2 -= lr*self.dL_db2
        self.W2 -= lr*self.dL_dW2
        self.b3 -= lr*self.dL_db3
        self.W3 -= lr*self.dL_dW3
        return 

class ReLU():
    def forward(self, x):
        output = np.where(x >= 0, x, 0)
        self.output = output
        self.derivative = np.where(x >= 0, 1, 0)
        return output
    
    def backward(self):
        return self.derivative

class MSloss():
    def __init__(self, ytrue):
        self.ytrue = ytrue
        self.value = float("inf")

    def forward(self, yhat):
        self.yhat = yhat
        self.value = np.dot((self.ytrue - self.yhat).T, (self.ytrue - self.yhat)) / self.yhat.shape[0]
        self.value = self.value.squeeze()
        return self.value
    
    def backward(self):
        return -2 * (self.ytrue - self.yhat) / self.yhat.shape[0]

def MSE(ytrue, yhat):
    return (np.dot((ytrue - yhat).T, (ytrue - yhat)) / yhat.shape[0]).squeeze()


data = sio.loadmat("bodyfat_data.mat")
X = data['X']
Y = data['y']
train_X, test_X = X[:150], X[150:]
train_Y, test_Y = Y[:150], Y[150:]
my_network = FCN(train_X.shape[1])
losses = []
eps = 1e-20
while len(losses) < 2 or losses[-2] - losses[-1] >= eps:
    loss = MSloss(train_Y)
    for x,y in zip(train_X,train_Y):
        my_network.forward(x)
        my_network.backward(y)
        my_network.update()
    yhat = my_network.forward(train_X)
    loss.forward(yhat)
    # print(loss.value)
    losses.append(loss.value)

train_Y_hat = my_network.forward(train_X)
train_loss = MSE(train_Y, train_Y_hat)
test_Y_hat = my_network.forward(test_X)
test_loss = MSE(test_Y, test_Y_hat)
print("Iteration number {}".format(len(losses)))
print("MSE on testing data: {}\nMSE on training data: {}".format(test_loss, train_loss))
plt.plot(range(1, 1 + len(losses)), losses)
plt.show()
