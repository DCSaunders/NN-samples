#!/usr/bin/env python
from __future__ import division
import argparse
import collections
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
    parser.add_argument('--file_in', help='File of vectors to fit')
    parser.add_argument('--load_proj', 
                        help='Location to load pickled input data and projected data from')
    parser.add_argument('--load_model',  
                        help='Location to load pickled model from')
    parser.add_argument('--save_input',
                        help='Location to save pickled input data')
    parser.add_argument('--save_model',
                        help='Location to save pickled model')
    parser.add_argument('--save_samples',
                        help='Location to save pickled samples from model')
    parser.add_argument('--save_labels',
                        help='Location to save labelled training samples')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Project samples/model into two dimensions using TSNE, and save a plot')
    parser.add_argument('--pickled_in', action='store_true', default=False,
                        help='If true, input vectors must be unpickled')
    parser.add_argument('--n_components', default=0,
                        help='Number of GMM components to fit')
    parser.add_argument('--full_covar', action='store_true', default=False,
                        help='Set if Gaussians have full, not diagonal, covariance')

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
    colors = cm.rainbow([i / gmix.n_components for i in gmix.predict(samples)])
    ax = plt.gca()
    ax.scatter(proj_samples[:,0], proj_samples[:,1], c=colors, alpha=0.8)
    plt.show()
    plt.savefig("fitted_classes.png")

def fit_samples(samples, n_components=0, full_covar=False):
    '''Fit GMM to samples
    Args: numpy array of sample coordinates'''
    # Fit a Gaussian mixture with EM
    print len(samples)
    samples_per_mode = 100
    if not n_components:
        # rule-of-thumb fitting
        n_components = int(len(samples) / samples_per_mode)
    print 'Fitting GMM with {} components'.format(n_components)
    if full_covar:
        gmm = mixture.GaussianMixture(n_components=n_components, 
                                      covariance_type='full')
    else:
        gmm = mixture.GaussianMixture(n_components=n_components, 
                                      covariance_type='diag')
    gmm.fit(samples)
    return gmm
    
def get_samples(args):
    proj_samples = None
    lengths = None
    if args.load_proj:
        with open(args.load_input, 'rb') as f:
            samples, proj_samples = cPickle.load(f)
            print 'Loaded {} pickled samples from {}'.format(
                len(samples), args.load_input)
    else:
        if args.pickled_in:
            samples = []
            lengths = []
            unpickler = cPickle.Unpickler(open(args.file_in, 'rb'))
            while True:
                try:
                    saved = unpickler.load()
                    sample = np.array(saved['states'][0][0], dtype=float)
                    samples.append(sample)
                    lengths.append(saved['length'])
                except (EOFError):
                    break
            print 'Unpickled {} samples from {}'.format(
                len(samples), args.file_in)
        else:
            samples = np.loadtxt(args.file_in)
            print 'Loaded samples file from {}'.format(args.file_in)
    if args.plot:
        if not proj_samples:
            proj_samples = tsne(samples, max_iter=200)
        plot_samples(proj_samples)
    return samples, proj_samples, lengths

def get_model(args):
    if args.file_in or args.load_proj:
        samples, proj_samples, lengths = get_samples(args)
    else:
        samples = proj_samples = lengths = None
    if args.load_model and not args.plot:
        with open(args.load_model, 'rb') as f:
            gmm = cPickle.load(f)
    else:
        gmm = fit_samples(samples, args.n_components, args.full_covar)
    return gmm, samples, proj_samples

def save_model(args, gmm, train_samples, proj_samples, label_samples):
    if args.save_input:
        with open(args.save_input, 'wb') as f:
            cPickle.dump([train_samples, proj_samples], f)
    if args.save_model:
        with open(args.save_model, 'wb') as f:
            cPickle.dump(gmm, f, protocol=-1)
    if args.save_samples:
        with open(args.save_samples, 'wb') as f:
            p = cPickle.Pickler(f)
            for sample in label_samples:
                p.dump(sample)

def sample_model(gmm, n_samples):
    samples, labels = gmm.sample(n_samples)
    label_samples = []
    scores = gmm.score_samples(samples)
    for label, sample, score in zip(labels, samples, scores):
        label_samples.append({'states': sample, 'label': label, 'score': score})
    return label_samples

def predict_model(gmm, samples, save_labels):
    labels = gmm.predict(samples)
    with open(save_labels, 'wb') as f:
        cPickle.dump(zip(samples, labels), f)

if __name__ == '__main__':
    args = get_args()
    gmm, train_samples, proj_samples = get_model(args)
    if args.save_labels:
        predict_model(gmm, train_samples[:100], args.save_labels)
    if args.plot:
        plot_clusters(train_samples, proj_samples, gmm)
    n_samples = 1000
    label_samples = sample_model(gmm, n_samples)
    save_model(args, gmm, train_samples, proj_samples, label_samples)
