from __future__ import division
import numpy as np
import argparse
import cPickle
import random


def unpickle_hidden(load_samples, max_in=0):
    hidden_list = []
    with open(load_samples, 'rb') as f_in:
        unpickler = cPickle.Unpickler(f_in)
        while True and (max_in == 0 or len(hidden_list) < max_in):
            try:
                hidden = unpickler.load()
                hidden_list.append(hidden)
            except (EOFError):
                break
    return hidden_list

def pickle_hidden(save_samples, hidden_list):
    with open(save_samples, 'wb') as f_out:
        pickler = cPickle.Pickler(f_out)
        for hidden in hidden_list:
            pickler.dump(hidden)

def main(load, save, max_in, scale):
    hidden_list = unpickle_hidden(load, max_in)
    for hidden in hidden_list:
        s = np.array(hidden['states'])
        #print np.mean(s), np.std(s)
        if scale > 0:
            noise_sample = np.random.normal(loc=0.0, scale=scale, size=s.size)
            hidden['states'] += noise_sample.reshape(s.shape)
    pickle_hidden(save, hidden_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_samples', type=str, default=None,
                        help='Location from which to load pickled samples')
    parser.add_argument('--save_samples', type=str, default=None,
                        help='Location to pickle predictions')
    parser.add_argument('--max_in', type=int, default=0,
                        help='Number of sentences to adjust')
    parser.add_argument('--scale', type=float, default=0.1,
                        help='Scale of noise to add')
    args = parser.parse_args()
    np.random.seed(1234)
    random.seed(1234)
    main(args.load_samples, args.save_samples, args.max_in, args.scale)
