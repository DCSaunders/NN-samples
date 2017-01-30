from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.models.rnn.translate.seq2seq.wrapper_cells import BidirectionalRNNCell, BOWCell
from tensorflow.python.ops import rnn, seq2seq
import matplotlib.pyplot as plt
import cPickle
np.random.seed(0)
tf.set_random_seed(0)

class VariationalAutoencoder(object):
    def __init__(self, dimensions, transfer_func=tf.nn.softplus, learning_rate=0.001, batch_size=100, use_lstm=True):
        self.weights = dict()
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.n_z = dimensions['n_z']
        self.transfer_func = transfer_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, dimensions["n_input"]]) 
        self._init_cell(dimensions, use_lstm)
        self._init_network(dimensions)
        self._init_optimizer()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def _init_cell(self, dimensions, use_lstm):
        self.cell = tf.nn.rnn_cell.LSTMCell(dimensions['n_h'], initializer=self.initializer)
        if use_lstm:
            dimensions['n_state'] = dimensions['n_h'] * 2
            print 'Using LSTMs: doubling hidden layer size to {}'.format(dimensions['n_state'])
        else:
            dimensions['n_state'] = dimensions['n_h']

    def _init_network(self, dimensions):
        self._init_weights('enc',
                           n_in=dimensions['n_input'],
                           n_h=dimensions['n_h'],
                           n_out=dimensions['n_z'])
        self._init_weights('dec',
                           n_in=dimensions['n_z'],
                           n_h=dimensions['n_h'],
                           n_out=dimensions['n_input'],
                           state_in=dimensions['n_state'])
        self.z_mean, self.z_log_var = self._enc_network()
        eps = tf.random_normal((self.batch_size, dimensions['n_z']),
                               0, 1, dtype=tf.float32)
        # reparameterization trick for encoder latent variable
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))
        self.recon_mean = self._dec_network()
    
    def _init_weights(self, name, n_in, n_h, n_out, state_in=0):
        self.weights[name] = dict(
            weights=dict(
                h=tf.Variable(self.initializer([n_in, n_h])),
                out_mean=tf.Variable(self.initializer([n_h, n_out])),
                out_log_var=tf.Variable(self.initializer([n_h, n_out]))),
            biases=dict(
                h=tf.Variable(tf.zeros([n_h], dtype=tf.float32)),
                out_mean=tf.Variable(tf.zeros([n_out], dtype=tf.float32)),
                out_log_var=tf.Variable(tf.zeros([n_out], dtype=tf.float32))))
        if state_in > 0:
             self.weights[name]['weights']['n_h_in'] = tf.Variable(
                 self.initializer([n_in, state_in]))
             self.weights[name]['biases']['n_h_in'] = tf.Variable(
                tf.zeros([state_in], dtype=tf.float32))


    def _rnn_in(self):
        return tf.split(split_dim=0, num_split=self.batch_size, value=self.x)


    def _enc_network(self):
        # Encoder mapping inputs onto Gaussian in latent space
        w, b = self.weights['enc']['weights'], self.weights['enc']['biases']
        _, enc_state = rnn.rnn(self.cell, self._rnn_in(), dtype=tf.float32)
        z_mean = tf.add(tf.matmul(enc_state, w['out_mean']), b['out_mean'])
        z_log_var = tf.add(tf.matmul(enc_state, w['out_log_var']), b['out_log_var'])
        return z_mean, z_log_var

    def _dec_network(self):
        # Decoder mapping latent space onto Bernoulli in data space
        w, b = self.weights['dec']['weights'], self.weights['dec']['biases']
        initial_state = self.transfer_func(tf.add(tf.matmul(self.z, w['n_h_in']), b['n_h_in']))
        dec_out, _ = seq2seq.rnn_decoder(self._rnn_in(), initial_state, self.cell)
        x_recon_mean = tf.nn.sigmoid(tf.add(tf.matmul(dec_out[-1], w['out_mean']), b['out_mean']))
        return x_recon_mean

    def _init_optimizer(self):
        # Find encoder (KL divergence) and decoder (E[P(X|z)]) loss
        encoder_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_var), 1)
        decoder_loss = -tf.reduce_sum(
                          self.x * tf.log(1e-10 + self.recon_mean) 
                        + (1 - self.x) * tf.log(1e-10 + (1 - self.recon_mean)), 1)
        self.cost = tf.reduce_mean(decoder_loss + encoder_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
       """Train model based on mini-batch of input data.
       Return cost of mini-batch.
       """
       _, cost = self.sess.run((self.optimizer, self.cost),
                               feed_dict={self.x: X})
       return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=(self.batch_size, self.n_z))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.recon_mean, feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.recon_mean, feed_dict={self.x: X})


def train(dimensions, learning_rate=0.001, batch_size=100,
          training_epochs=10, display_step=1):
    vae = VariationalAutoencoder(dimensions, learning_rate=learning_rate,
                                 batch_size=batch_size)
    for epoch in range(training_epochs):
        avg_cost = 0.0
        batch_count = int(n_samples / batch_size)
        for i in range(batch_count):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            cost = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            print "Epoch: {:04d} cost={:.9f}".format(epoch + 1, avg_cost)
    return vae

def plot_reconstruct(x_sample, x_reconstruct):
    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.show()

def plot_generate(vae, view_count=10, save=None):
    fig, ax = plt.subplots(1, view_count, figsize=(view_count, 2))
    fig.suptitle('Samples from VAE')
    gen = vae.generate() 
    for i in range(view_count):  
        ax[i].imshow(gen[i].reshape(28, 28), vmin=0, vmax=1)
    plt.show()
    if save:
        with open(save, 'wb') as f_out:
            cPickle.dump(gen, f_out)

dimensions = dict(n_h=300, # RNN cell hidden layer size
                  n_input=784, # MNIST data input (img shape: 28*28)
                  n_z=20)  # Latent variable dimensionality
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
np.random.seed(1234)
n_samples = mnist.train.num_examples
batch_size = 100
vae = train(dimensions, training_epochs=20, batch_size=batch_size)
x_sample = mnist.test.next_batch(batch_size)[0]
#x_reconstruct = vae.reconstruct(x_sample)
#plot_reconstruct(x_sample, x_reconstruct)
#plot_generate(vae, save='/home/mifs/ds636/exps/mnist/vae/sample_reconstructions')
