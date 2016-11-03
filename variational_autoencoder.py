# Walkthrough and tweaks to Variational Autoencoder in Tensorflow post by Jan Hendrik Metzen https://jmetzen.github.io/2015-11-27/vae.html

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
np.random.seed(0)
tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    """Xavier initialization of network weights - depends on number of incoming and outgoing connections.
    cf Glorot and Bengio http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

class VariationalAutoencoder(object):
    def __init__(self, dimensions, f_transfer=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.dimensions = dimensions
        self.f_transfer = f_transfer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # input
        self.x = tf.placeholder(tf.float32, [None, dimensions["n_input"]]) 
        self._init_network()
        self._init_optimizer()
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _init_network(self):
        weights = self._init_weights(**self.dimensions)
        self.z_mean, self.z_log_var = self._latent_network(
            weights["weights_latent"], weights["biases_latent"])
        n_z = self.dimensions["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)
        # reparameterization trick for latent variable
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))
        self.reconstruct_mean = self._generator(
            weights["weights_gener"], weights["biases_gener"])
    
    def _init_weights(self, n_hidden_latent_1, n_hidden_latent_2, n_hidden_gener_1, n_hidden_gener_2, n_input, n_z):
        all_weights = dict()
        all_weights['weights_latent'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_latent_1)),
            'h2': tf.Variable(xavier_init(n_hidden_latent_1, n_hidden_latent_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_latent_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_latent_2, n_z))}
        all_weights['biases_latent'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_latent_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_latent_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _latent_network(self, weights, biases):
        # Probabilistic encoder mapping inputs onto Gaussian in latent space
        layer_1 = self.f_transfer(
            tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.f_transfer(
            tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        z_mean = tf.add(tf.matmul(
            layer_2, weights['out_mean']), biases['out_mean'])
        z_log_var = tf.add(tf.matmul(
            layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        return z_mean, z_log_var

    def _generator(self, weights, biases):
        # Probabilistic decoder mapping latent space onto Bernoulli in 
        # data space
        layer_1 = self.f_transfer(
            tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.f_transfer(
            tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        x_reconstr_mean = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))
        return x_reconstr_mean

    
    def _init_optimizer(self):
        # E[P(X|z)] loss
        decoder_loss = -tf.reduce_sum(
            self.x * tf.log(1e-10 + self.reconstruct_mean) 
            + (1 - self.x) * tf.log(1e-10 + (1 - self.reconstruct_mean)), 1)
        # KL divergence loss
        encoder_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_var), 1)
        # Average over batch
        self.cost = tf.reduce_mean(decoder_loss + encoder_loss)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
       """Train model based on mini-batch of input data.
       Return cost of mini-batch.
       """
       opt, cost = self.sess.run((self.optimizer, self.cost), 
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
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.dimensions["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.reconstruct_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.reconstruct_mean, 
                             feed_dict={self.x: X})



def train(dimensions, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(dimensions, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        batch_count = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(batch_count):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae

network_architecture = dict(n_hidden_latent_1=500, # 1st layer encoder neurons
                            n_hidden_latent_2=500, # 2nd layer encoder neurons
                            n_hidden_gener_1=500, # 1st layer decoder neurons
                            n_hidden_gener_2=500, # 2nd layer decoder neurons
                            n_input=784, # MNIST data input (img shape: 28*28)
                            n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=15)
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)
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
plt.tight_layout()
