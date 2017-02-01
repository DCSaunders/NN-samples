from __future__ import division
import numpy as np
import cPickle
f_name = '/home/mifs/ds636/exps/mnist/simple_classifier/reconstruct_predicts_vae'
with open(f_name, 'rb') as f:
    r = cPickle.load(f)
predictions = np.asarray([item[1] for item in r])
labels = np.asarray([np.argmax(arr) for arr in predictions])
conf = np.zeros(len(labels))
for i, arr in enumerate(predictions):
    conf[i] = arr[labels[i]]
print 'mean certainty in label: {}'.format(np.mean(conf))
print 'stddev of certainties: {}'.format(np.std(conf))
print 'proportion of certainties below 0.8: {} min certainty: {}'.format(
    len(conf[np.where(conf < 0.8)]) / len(labels), min(conf))
