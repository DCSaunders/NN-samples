#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from scipy.stats import multivariate_normal
from tsne import tsne

def plot_samples(samples):
    '''Plot chosen samples
    Args:
    samples: numpy array of sample coordinates'''
    ax = plt.gca()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.savefig("samples.png")
    plt.clf()
    
def fit_samples(samples, proj_samples):
    '''Fit GMM to samples
    Args: numpy array of sample coordinates'''
    gmix = mixture.GMM(n_components=2, covariance_type='full')
    gmix.fit(samples)
    print gmix.means
    colors = cm.rainbow(np.array([i for i in gmix.predict(samples)]))
    ax = plt.gca()
    ax.scatter(proj_samples[:,0], proj_samples[:,1], c=colors, alpha=0.8)
    plt.savefig("fitted_classes.png")
 
if __name__ == '__main__':
    samples = np.loadtxt("mnist2500_X.txt")
    proj_samples = tsne(samples)
    plot_samples(proj_samples)
    fit_samples(samples, proj_samples)
