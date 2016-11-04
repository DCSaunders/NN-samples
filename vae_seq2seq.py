# Attempt to change previous VAE work to handle seq2seq data
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
np.random.seed(0)
tf.set_random_seed(0)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class VariationalAutoencoder(object):
    def __init__(self, dimensions, transfer_func=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.weights = dict()
        self.transfer_func = transfer_func
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.x = tf.placeholder(tf.float32, [None, dimensions["n_input"]]) 
        self._init_network(dimensions)
        self._init_optimizer()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def _init_network(self, dimensions):
        self._init_weights('enc', n_in=dimensions['n_input'],  n_h_1=dimensions['n_h_enc_1'],
                           n_h_2=dimensions['n_h_enc_2'], n_out=dimensions['n_z'])
        self._init_weights('dec', n_in=dimensions['n_z'], n_h_1=dimensions['n_h_dec_1'],
                           n_h_2=dimensions['n_h_dec_2'],  n_out=dimensions['n_input'])
        self.z_mean, self.z_log_var = self._enc_network()
        eps = tf.random_normal((self.batch_size, dimensions['n_z']), 0, 1, dtype=tf.float32)
        # reparameterization trick for encoder latent variable
        self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))
        self.reconstruct_mean = self._dec_network()
    
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
        layer_1 = self.transfer_func(tf.add(tf.matmul(self.z, w['h1']), b['h1']))
        layer_2 = self.transfer_func(tf.add(tf.matmul(layer_1, w['h2']), b['h2']))
        x_reconstruct_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w['out_mean']), b['out_mean']))
        return x_reconstruct_mean

    def _init_optimizer(self):
        # Find encoder (KL divergence) and decoder (E[P(X|z)]) loss
        encoder_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_var), 1)
        decoder_loss = -tf.reduce_sum(
                          self.x * tf.log(1e-10 + self.reconstruct_mean) 
                        + (1 - self.x) * tf.log(1e-10 + (1 - self.reconstruct_mean)), 1)
        self.cost = tf.reduce_mean(decoder_loss + encoder_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
       """Train model based on mini-batch of input data.
       Return cost of mini-batch.
       """
       _, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
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
            z_mu = np.random.normal(size=self.dimensions["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.reconstruct_mean, feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.reconstruct_mean, feed_dict={self.x: X})


def train(dimensions, learning_rate=0.001, batch_size=100, training_epochs=10, display_step=1):
    vae = VariationalAutoencoder(dimensions, learning_rate=learning_rate, batch_size=batch_size)
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



learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



'''
dimensions = dict(n_h_enc_1=300, # Encoder 1st hidden layer size
                  n_h_enc_2=300, # Encoder 2nd hidden layer size
                  n_h_dec_1=300, # Decoder 1st hidden layer size
                  n_h_dec_2=300, # Decoder 2st hidden layer size
                  n_input=784, # MNIST data input (img shape: 28*28)
                  n_z=20)  # Latent variable dimensionality
n_samples = mnist.train.num_examples
batch_size = 100
vae = train(dimensions, training_epochs=1, batch_size=batch_size)
x_sample = mnist.test.next_batch(batch_size)[0]
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

'''
