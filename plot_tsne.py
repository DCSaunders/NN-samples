from __future__ import division

import cPickle
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from tsne import tsne
f_name = '/home/mifs/ds636/exps/mnist/simple_classifier/reconstruct_predicts_vae'
np.random.seed(1234)
with open(f_name, 'rb') as f:
    r = cPickle.load(f)
predictions = np.asarray([item[1] for item in r])
labels = np.asarray([np.argmax(arr) for arr in predictions])
Y = tsne(predictions, 2, 50, 20.0, max_iter=200)

# Define a colormap with the right number of colors
cmap = plt.cm.get_cmap('jet', 10)
bounds = range(0,11)
norm = colors.BoundaryNorm(bounds, cmap.N)
fig, ax = plt.subplots()
scat = ax.scatter(Y[:,0], Y[:,1], c=labels, s=50, cmap=cmap, norm=norm)
# Add a colorbar. Move the ticks up by 0.5, so they are centred on the colour.
cb = fig.colorbar(scat)
N = 10
cols = np.arange(0,N,1)
cb.set_ticks(cols + 0.5)
cb.set_ticklabels(cols)
plt.show()

