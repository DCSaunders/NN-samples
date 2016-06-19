#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from random import sample

# Evaluate total loss (cross-entropy)
def calculate_loss(model):
    # Weights between input and hidden layer
    W1, b1 = model['W1'], model['b1']
    # Weights between hidden layer and output
    W2, b2 = model['W2'], model['b2']
    # Forward propagation to calculate predictions
    z1 = X.dot(W1) + b1
    # tanh activation function at hidden layer
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    # softmax activation function at output layer
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calulate loss
    true_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(true_logprobs)
    # Add regularization term
    data_loss +=reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# Predict a NN output (0 or 1)
def predict(mode, x):
    # Weights between input and hidden layer
    W1, b1 = model['W1'], model['b1']
    # Weights between hidden layer and output
    W2, b2 = model['W2'], model['b2']
    #Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True)
    return np.argmax(probs, axis=1)

def minibatch(X, y, size):
    x_sub, y_sub = zip(*sample(list(zip(X, y)), size))
    return np.array(x_sub), np.array(y_sub)

# Learn NN parameters and return model
# nn_hdim: number of hidden layer nodes
# num_passes: passes through training data for grad desc
# print_loss: if True, print loss every 1000 iterations
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    epsilon = 0.01  # learning rate for gradient descent
    # Initialise parameters to random values
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1,nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1,nn_output_dim))
    model = {} # to return
    # Gradient descent: iterate through batches
    for ii in xrange(0, num_passes):
        #forward propagation
        batchsize = 16
        Xbatch, ybatch = minibatch(X, y, batchsize)
        z1 = Xbatch.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims = True)
        # back propagation
        delta3 = probs
        delta3[range(batchsize), ybatch] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T)*(1-np.power(a1,2)) # from tanh derivative
        dW1 = np.dot(Xbatch.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # Add regularization terms to weights W
        dW2 += reg_lambda*W2
        dW1 += reg_lambda*W1
        # Gradient descent parameter update
        W1+= -epsilon*dW1
        b1+= -epsilon*db1
        W2+= -epsilon*dW2
        b2+= -epsilon*db2

        if ii%100 == 0:
            epsilon = 0.9*epsilon
        model = {'W1': W1,'b1': b1,'W2': W2,'b2': b2}
        if print_loss and ii%1000 == 0:
            print "Loss after iteration %i: %f" %(ii, calculate_loss(model))
    return model         


# Generates contour plot for decision boundary
def plot_decision_boundary(pred_func):
    # Set max and min values, and padding
    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    h = 0.01
    # Generate grid of points spaced by distance h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:, 1], c=y, cmap = plt.cm.Spectral)
    
# Scatter graph of data
def plotScatter():
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.title("Dataset")
    plt.show()    

# Fit logistic regression
def plotLR():
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X,y)
    plot_decision_boundary(lambda x: clf.predict(x))
    plt.title("Logistic regression")
    plt.show()

    
# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)

# Gradient descent parameters handpicked
reg_lambda = 0.01  # regularization strength


# Create dataset
np.random.seed(0)
X, y = sklearn.datasets.make_blobs(n_samples=300, centers=3)

#X, y = sklearn.datasets.make_moons(200, noise=0.20) #2-class dataset
num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dim
nn_output_dim = 3 # output layer dim: num classes
#plotScatter()
#plotLR()
for num_hidden in range (3, 10):
    model = build_model(X, y, num_hidden, num_passes=10000, print_loss=True)
    plot_decision_boundary(lambda x: predict(model, x))
    t = "Decision boundary for hidden layer size %s" % num_hidden
    plt.title(t)
    plt.show()

