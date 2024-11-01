#PART 1 

#importing libraries
import numpy as np

W = np.random.randn(1, 2)
B = np.random.randn(1)

#checking randm value of B (bias)
#print(B)

#creating first neuron
def sigm(X, W, B):
    M = 1/(1+np.exp(-(X.dot(W.T)+ B)))
    return M



#derive analytical expression




#update rules and weights (W) for bias (B)
def diff_W(X, Z, Y, B, W):
    dS = sigm(X, W, B) * (1-sigm(X, W, B))
    dW = (Z-Y)*dS
    return X.T.dot(dW)


def diff_B(X, Z, Y, B, W):
    dS = sigm(X, W, B) * (1-sigm(X, W, B))
    dB = (Z-Y)*dS
    return dB.sum(axis = 0)


#train and test set

#Train
#creating 15 2 dimensional samples
X = np.random.randit(2, size = [15, 2])
Y = np.array([X[:,0] | X[:,1]]).T



#Test
X_Test = np.random.randit(2, size = [15, 2])
Y_Test = np.array([X[:,0] | X[:,1]]).T


#now we can teach vthe neuron to emulate the OR func
learning_rate = 0.01

for epoch in range(500):
    output = sigm(X, W, B)
    W -= learning_rate * diff_W(W, output, Y, B, W).T
    B -= learning_rate * diff_B(W, output, Y, B, W)
    error_1 = np.mean(( Y-output) **2)
    accuracy = 1 - error_1
    print(accuracy)

#and repeating the same for the AND func


