#PART 1 

#importing libraries
import numpy as np

W = np.random.randn(1, 2)
B = np.random.randn(1)

#checking randm value of B (bias)
#print(B)

#creating first neuron
def sigmoid(X, W, B):
    M = 1/(1+np.exp(-(X.dot(W.T)+ B)))
    return M


# #derive analytical expression - in diff_w/b funcs!!!
# def sigmoid_derivative(X, W, B):
#     S = sigmoid(X, W, B)
#     return S * (1 - S)


#update rules and weights (W) for bias (B)
def diff_W(X, Z, Y, B, W):
    dS = sigmoid(X, W, B) * (1-sigmoid(X, W, B))
    dW = (Z-Y)*dS
    return X.T.dot(dW)


def diff_B(X, Z, Y, B, W):
    dS = sigmoid(X, W, B) * (1-sigmoid(X, W, B))
    dB = (Z-Y)*dS
    return dB.sum(axis = 0)


#train and test set::
#Train
#creating 15 2 dimensional samples
X = np.random.randint(2, size = [15, 2])
Y_OR = np.array([X[:,0] | X[:,1]]).T
Y_AND = np.array([X[:,0] & X[:,1]]).T


#Test
X_Test = np.random.randint(2, size = [15, 2])
Y_Test_OR = np.array([X[:, 0] | X_Test[:, 1]]).T
Y_Test_AND = np.array([X[:, 0] & X_Test[:, 1]]).T


#now we can teach the neuron to emulate the OR func
learning_rate = 0.01

print("Training Neuron for OR Function")
for epoch in range(500):
    output = sigmoid(X, W, B)
    W -= learning_rate * diff_W(X, output, Y_OR, B, W).T
    B -= learning_rate * diff_B(X, output, Y_OR, B, W)
    error_1 = np.mean((Y_OR - output) **2)
    accuracy = 1 - error_1
    print("OR func accuacy: ", accuracy)


#and repeating the same for the AND func (same learning rate)
print("Training Neuron for AND Function")
for epoch in range(500):
    output = sigmoid(X, W, B)
    W -= learning_rate * diff_W(X, output, Y_AND, B, W).T
    B -= learning_rate * diff_B(X, output, Y_AND, B, W)
    error_2 = np.mean((Y_AND - output) **2)
    accuracy = 1 - error_2
    print("AND func accuacy: ", accuracy)