#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
 
def true_mixture(x, y):
    '''
    Define multivariate normal distributions, and return a weighted
    sum of samples from them.
    Args:
    x: x coordinate to sample from distributions
    y: y coordinate to sample from distributions
    Returns:
    A weighted sum of MVN samples.'''
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6*g1+28.4*g2/(0.6+28.4)
 
def plot_data():
    '''
    Defines a grid of samples and plots a 3D map of the probability 
    distribution defined by a given GMM.'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = true_mixture(X, Y)
    colours=plt.get_cmap('coolwarm')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=colours, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('true_model.png')
    plt.clf()
 
def sample(iters):
    '''Perform Metropolis Hastings sampling to estimate the model
    Args:
    iters: int number of iterations for Metropolis Hastings sampling
    Returns:
    samples: numpy array of chosen useful coordinates to sample with.
    '''
    spacing = 10
    r = np.zeros(2)
    p = true_mixture(r[0], r[1]) # some probability
    samples = []
    for index in xrange(iters):
	random_sample = r + np.random.normal(size=2)
	sample_output = true_mixture(random_sample[0], random_sample[1])
	if sample_output >= p:
	    p = sample_output
	    r = random_sample
	else:
	    random_val = np.random.rand()
	    if random_val < sample_output/p:
		p = sample_output
		r = random_sample
	if index % spacing == 0:
	    samples.append(r) 
    samples = np.array(samples)
    return samples

def plot_samples(samples):
    '''Plot chosen samples
    Args:
    samples: numpy array of sample coordinates'''
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
    dx = 0.01
    x = np.arange(np.min(samples), np.max(samples), dx)
    y = np.arange(np.min(samples), np.max(samples), dx)
    X, Y = np.meshgrid(x, y)
    Z = true_mixture(X, Y)
    CS = plt.contour(X, Y, Z, 10, alpha=0.5)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig("samples.png")

def fit_samples(samples):
    '''Fit GMM to samples
    Args: numpy array of sample coordinates
    '''
    gmix = mixture.GMM(n_components=2, covariance_type='full')
    gmix.fit(samples)
    print gmix.means_
    colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
    ax = plt.gca()
    ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
    plt.savefig("fitted_classes.png")
 
if __name__ == '__main__':
    iters = 10000
    plot_data()
    samples = sample(iters)
    plot_samples(samples)
    fit_samples(samples)
