#!/usr/bin/env python
from __future__ import division
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
import cPickle
from sklearn import mixture
from scipy.stats import multivariate_normal
from tsne import tsne

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', '-i',
                        help='File of vectors to fit')
    parser.add_argument('--load_samples', 
                        help='Location to load pickled samples')
    parser.add_argument('--load_model',  
                        help='Location to load pickled model')
    parser.add_argument('--save_samples',
                        help='Location to save pickled samples')
    parser.add_argument('--save_model', 
                        help='Location to save pickled model')
    return parser.parse_args()

def plot_samples(samples):
    '''Plot chosen samples
    Args:
    samples: numpy array of sample coordinates'''
    ax = plt.gca()
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.savefig("samples.png")
    plt.clf()

def plot_clusters(samples, proj_samples, gmix):
    colors = cm.rainbow([i / gmix.n_components 
                         for i in gmix.predict(samples)])
    ax = plt.gca()
    ax.scatter(proj_samples[:,0],
               proj_samples[:,1], c=colors, alpha=0.8)
    plt.show()
    plt.savefig("fitted_classes.png")


def fit_samples(samples):
    '''Fit GMM to samples
    Args: numpy array of sample coordinates'''
    bic = []
    best_gmm = None
    lowest_bic = np.infty
    n_components_range = [10]
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components)
        gmm.fit(samples)
        bic.append(gmm.bic(samples))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
            print 'best model so far: n_components={}'.format(n_components)
    return best_gmm
    
if __name__ == '__main__':
    args = get_args()
    if args.load_model:
        with open(args.load_model, 'rb') as f:
            gmm = cPickle.load(f)
    else:
        if args.load_samples:
            with open(args.load_samples, 'rb') as f:
                samples, proj_samples = cPickle.load(f)
            print 'Loaded pickled samples from {}'.format(args.load_samples)
        else:
            samples = np.loadtxt(args.file_in)
            print 'Loaded samples file from {}'.format(args.file_in)
            #proj_samples = tsne(samples, max_iter=200)
        if args.save_samples:
            with open(args.save_samples, 'wb') as f:
                cPickle.dump([samples, proj_samples], f)
        #plot_samples(proj_samples)
        gmm = fit_samples(samples)
    #plot_clusters(samples, proj_samples, gmm)
    print gmm.sample()
    if args.save_model:
        with open(args.save_model, 'wb') as f:
            cPickle.dump(gmm, f)    
