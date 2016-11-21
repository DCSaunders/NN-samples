#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from scipy.stats import multivariate_normal
 
def proposal(sample):
    '''
    Define multivariate normal distributions, and return a weighted
    sum of samples from them.
    Args:
    sample: array of samples to feed into weighted MVN distributions
    Returns:
    A weighted sum of MVN samples.'''
    mean_1 = [-1.0, -1.0]
    cov_1 = [[1.0, -0.8], [-0.8, 1.0]]
    mean_2 = [1, 2]
    cov_2 = [[1.5, 0.6], [0.6, 0.8]]
    g1 = multivariate_normal.pdf(sample, mean_1, cov_1)
    g2 = multivariate_normal.pdf(sample, mean_2, cov_2)
    return 0.6*g1+28.4*g2/(0.6+28.4)
 
def plot_data():
    '''
    Defines a grid of samples and plots a 3D map of the probability 
    distribution defined by a given GMM.'''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    plot_x, plot_y, plot_z = regrid(x, y)
    colours=plt.get_cmap('coolwarm')
    surf = ax.plot_surface(plot_x, plot_y, plot_z, rstride=1, cstride=1,
                           cmap=colours, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('true_model.png')
    plt.clf()

def regrid(x, y):
    plot_x, plot_y = np.meshgrid(x, y)
    Z = proposal(zip(plot_x.flatten(), plot_y.flatten()))
    plot_z = np.reshape(Z, np.shape(plot_x))
    return plot_x, plot_y, plot_z
    
def sample(iters, d):
    '''Perform Metropolis Hastings sampling to estimate the model
    Args:
    iters: int number of iterations for Metropolis Hastings sampling
    d: int number of dimensions per (1D) sample vector
    Returns:
    samples: numpy array of chosen useful coordinates to sample with.
    '''
    spacing = 10
    accepted_sample = np.zeros(d)
    current_state = proposal(accepted_sample) 
    samples = []
    for index in xrange(iters):
	current_sample = accepted_sample + np.random.normal(size=d)
	new_state = proposal(current_sample)
        # If the new state is more probable, we always accept
	if new_state >= current_state: 
	    current_state = new_state
	    accepted_sample = current_sample
	else:
            # If the new state is less probable, we may still accept
	    accept_prob = np.random.rand()
	    if accept_prob < new_state/current_state:
		current_state = new_state
		accepted_sample = current_sample
	if index % spacing == 0:
	    samples.append(accepted_sample) 
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
    plot_x, plot_y, plot_z = regrid(x, y)
    CS = plt.contour(plot_x, plot_y, plot_z, 10, alpha=0.5)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig("samples.png")

def fit_samples(samples):
    '''Fit GMM to samples
    Args: numpy array of sample coordinates'''
    gmix = mixture.GMM(n_components=2, covariance_type='full')
    gmix.fit(samples)
    print gmix.means_
    colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
    ax = plt.gca()
    ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.8)
    plt.savefig("fitted_classes.png")
 
if __name__ == '__main__':
    iters = 10000
    dimensions = 2
    plot_data()
    samples = sample(iters, dimensions)
    plot_samples(samples)
    fit_samples(samples)
