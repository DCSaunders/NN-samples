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
    parser.add_argument('--load_input', 
                        help='Location to load pickled input data from')
    parser.add_argument('--load_model',  
                        help='Location to load pickled model from')
    parser.add_argument('--save_input',
                        help='Location to save pickled input data')
    parser.add_argument('--save_model',
                        help='Location to save pickled model')
    parser.add_argument('--save_samples',
                        help='Location to save pickled samples from model')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Project samples/model into two dimensions using TSNE, and save a plot')
    parser.add_argument('--pickled_in', action='store_true', default=False,
                        help='If true, input vectors must be unpickled')

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
    
def get_samples(args):
    proj_samples = None
    if args.load_input:
        with open(args.load_input, 'rb') as f:
            samples, proj_samples = cPickle.load(f)
            print 'Loaded pickled samples from {}'.format(args.load_input)
    else:
        if args.pickled_in:
            samples = []
            unpickler = cPickle.Unpickler(open(args.file_in, 'rb'))
            while True:
                try:
                    sample = np.array(unpickler.load()[0][0], dtype=float)
                    samples.append(sample)
                except (EOFError):
                    break
            print 'Unpickled samples from {}'.format(args.file_in)
        else:
            samples = np.loadtxt(args.file_in)
            print 'Loaded samples file from {}'.format(args.file_in)
    if args.plot:
        if not proj_samples:
            proj_samples = tsne(samples, max_iter=200)
        plot_samples(proj_samples)
    return samples, proj_samples

def get_model(args):
    if args.load_model and not args.plot:
        with open(args.load_model, 'rb') as f:
            gmm = cPickle.load(f)
        samples = proj_samples = None
    else:
        samples, proj_samples = get_samples(args)
        gmm = fit_samples(samples)
    return gmm, samples, proj_samples

def save_model(args, gmm, train_samples, proj_samples,
               label_sample_dict, label_score_dict):
    if args.save_input:
        with open(args.save_input, 'wb') as f:
            cPickle.dump([train_samples, proj_samples], f)
    if args.save_model:
        with open(args.save_model, 'wb') as f:
            cPickle.dump(gmm, f)
    if args.save_samples:
        with open(args.save_samples, 'wb') as f:
            cPickle.dump([label_sample_dict, label_score_dict], f)

def sample_model(gmm, n_samples):
    samples, labels = gmm.sample(n_samples)
    label_sample_dict = collections.defaultdict(list)
    label_score_dict = collections.defaultdict(list)
    #responsibilities = gmm.predict_proba(samples)
    scores = gmm.score_samples(samples)
    for label, sample, score in zip(
            labels, samples, scores):
        label_sample_dict[label].append(sample)
        label_score_dict[label].append(score)
    return label_sample_dict, label_score_dict

if __name__ == '__main__':
    args = get_args()
    gmm, train_samples, proj_samples = get_model(args)
    if args.plot:
        plot_clusters(train_samples, proj_samples, gmm)
    n_samples = 1000
    label_sample_dict, label_score_dict = sample_model(gmm, n_samples)
    save_model(args, gmm, train_samples, proj_samples,
               label_sample_dict, label_score_dict)
