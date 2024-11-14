#PART 2 

#importing libraries
import numpy as np


#updating weights and biases 
W1 = np.random.randn(3, 2)
W2= np.random.randn(1, 3)
B1 = np.random.randn(3)
B2 = np.random.randn(1)

#checking randm value of B (bias)
#print(B)

#creating signmoid func for weights and biases
def sigmoid(X, W, B):
    M = 1/(1+np.exp(-(X.dot(W.T)+ B)))
    return M

#feedforward network 
def Forward(X, W1, B1, W2, B2):
    H = sigmoid(X, W1, B1)

    Y = sigmoid(X, W2, B2)

    return Y, H



#update rules and weights (W) for bias (B)
def diff_W1(X, H, Z, Y, W2):
    dZ = (Y-Z).dot(W2)*Y*(1-Y)*H*(1-H)
    return X.T.dot(dZ)

def diff_W2(H, Z, Y):
    dW = (Y-Z)*Y*(1-Y)
    return H.T.dot(dW)


def diff_B1(H, Z, Y, W2):
    return ((Y-Z).dot(W2)*Y*(1-Y)*H*(1-H))


def diff_B2(Z, Y):
    dB = (Z-Y)*Y*(1-Y)
    return dB.sum(axis = 0)


#train and test set

#Train
#creating 15 2 dimensional samples
X = np.random.randint(2, size = [15, 2])
Y = np.array([X[:,0] | X[:,1]]).T



#Test
X_Test = np.random.randint(2, size = [15, 2])
Y_Test = np.array([X[:,0] | X[:,1]]).T


#now we can teach vthe neuron to emulate the OR func
learning_rate = 0.01

for epoch in range(10000):
    Y, H = Forward(X, W1, B1, W2, B2)

    W1 -= learning_rate * diff_W1(X, H, Z, Y, W2).T
    W2 -= learning_rate * diff_W2(H, Z, Y).T
    B1 -= learning_rate * diff_B1(Z, Y, W2, H)
    B2 -= learning_rate * diff_B2(Z, Y)
    
    if not epoch % 50:
        accuracy = 1 - np.mean((Z-Y)**2)
        print('Epoch: ', epoch, ' Accuracy: ', accuracy)

#and repeating the same for the AND func






