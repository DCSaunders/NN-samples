# Walkthrough and refactoring to Variational Autoencoder in Tensorflow post by Jan Hendrik Metzen 
# (https://jmetzen.github.io/2015-11-27/vae.html)

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
np.random.seed(0)
tf.set_random_seed(0)

class VariationalAutoencoder(object):
    def __init__(self, dimensions, transfer_func=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.weights = dict()
        self.n_z = dimensions['n_z']
        self.transfer_func = transfer_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, dimensions["n_input"]]) 
        self._init_network(dimensions)
        self._init_optimizer()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def _init_network(self, dimensions):
        self._init_weights('enc',
                           n_in=dimensions['n_input'],
                           n_h_1=dimensions['n_h_enc_1'],
                           n_h_2=dimensions['n_h_enc_2'],
                           n_out=dimensions['n_z'])
        self._init_weights('dec',
                           n_in=dimensions['n_z'],
                           n_h_1=dimensions['n_h_dec_1'],
                           n_h_2=dimensions['n_h_dec_2'],
                           n_out=dimensions['n_input'])
        self.z_mean, self.z_log_var = self._enc_network()
        eps = tf.random_normal((self.batch_size, dimensions['n_z']),
                               0, 1, dtype=tf.float32)
        # reparameterization trick for encoder latent variable
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))
        self.recon_mean = self._dec_network()
    
    def _init_weights(self, name, n_in, n_h_1, n_h_2, n_out):
        xavier = tf.contrib.layers.xavier_initializer()
        self.weights[name] = dict(
            weights=dict(
                h1=tf.Variable(xavier([n_in, n_h_1])),
                h2=tf.Variable(xavier([n_h_1, n_h_2])),
                out_mean=tf.Variable(xavier([n_h_2, n_out])),
                out_log_var=tf.Variable(xavier([n_h_2, n_out]))),
            biases=dict(
                h1=tf.Variable(tf.zeros([n_h_1], dtype=tf.float32)),
                h2=tf.Variable(tf.zeros([n_h_2], dtype=tf.float32)),
                out_mean=tf.Variable(tf.zeros([n_out], dtype=tf.float32)),
                out_log_var=tf.Variable(tf.zeros([n_out], dtype=tf.float32))))

    def _enc_network(self):
        # Encoder mapping inputs onto Gaussian in latent space
        w, b = self.weights['enc']['weights'], self.weights['enc']['biases']
        layer_1 = self.transfer_func(tf.add(tf.matmul(self.x, w['h1']), b['h1']))
        layer_2 = self.transfer_func(tf.add(tf.matmul(layer_1, w['h2']), b['h2']))
        z_mean = tf.add(tf.matmul(layer_2, w['out_mean']), b['out_mean'])
        z_log_var = tf.add(tf.matmul(layer_2, w['out_log_var']), b['out_log_var'])
        return z_mean, z_log_var

    def _dec_network(self):
        # Decoder mapping latent space onto Bernoulli in data space
        w, b = self.weights['dec']['weights'], self.weights['dec']['biases']
        layer_1 = self.transfer_func(tf.add(tf.matmul(self.z, w['h1']),
                                            b['h1']))
        layer_2 = self.transfer_func(tf.add(tf.matmul(layer_1, w['h2']),
                                            b['h2']))
        x_recon_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w['out_mean']),
                                            b['out_mean']))
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
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.cost)

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
            z_mu = np.random.normal(size=(1, self.n_z))
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

def plot_generate(vae, reconstruct_count=10):
    fig, ax = plt.subplots(1, reconstruct_count, figsize=(reconstruct_count, 2))
    fig.suptitle('Samples from VAE') 
    for i in range(reconstruct_count):
        gen = vae.generate()
        ax[i].imshow(gen.reshape(28, 28), vmin=0, vmax=1)
    plt.show()

dimensions = dict(n_h_enc_1=300, # Encoder 1st hidden layer size
                  n_h_enc_2=300, # Encoder 2nd hidden layer size
                  n_h_dec_1=300, # Decoder 1st hidden layer size
                  n_h_dec_2=300, # Decoder 2st hidden layer size
                  n_input=784, # MNIST data input (img shape: 28*28)
                  n_z=20)  # Latent variable dimensionality
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples
batch_size = 100
vae = train(dimensions, training_epochs=1, batch_size=batch_size)
x_sample = mnist.test.next_batch(batch_size)[0]
x_reconstruct = vae.reconstruct(x_sample)
plot_reconstruct(x_sample, x_reconstruct)
#plot_generate(vae)
