import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.ops import rnn
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

class LSTM_VAE(object):
  def __init__(self, n_hidden, inputs, batch_size, elem_num, step_num, cell,
               learning_rate=0.001, reverse=True, n_latent=10, transfer_func=tf.nn.softplus,
               initializer=tf.truncated_normal, decode_with_input=False):
    """
    n_hidden : number of hidden elements of each LSTM unit.
    inputs : a list of input tensors with size (batch_size x elem_num)
    cell : an rnn cell object
    learning_rate: learning rate for Adam optimizer
    reverse : Option to decode in reverse order.
    decode_without_input : Option to decode without input.
    n_latent: number of latent states
    transfer_func: function on latent state to give decoder initial state
    decode_with_input: true if feeding decoder outputs back into decoder
    """
    self.batch_size = batch_size
    self.elem_num = elem_num
    self.step_num = step_num
    self.n_hidden = n_hidden
    self.n_latent = n_latent
    self._enc_cell = cell
    self._dec_cell = cell
    self.initializer = initializer
    self.transfer_func = transfer_func
    self.learning_rate = learning_rate
    self.init_reverse = reverse
    self.decode_with_input = decode_with_input
    self.build_network(inputs)

  def build_network(self, inputs):
    with tf.variable_scope('encoder'):
      z_mean_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent]))
      z_mean_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))
      z_logvar_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent]))
      z_logvar_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))

      _, enc_state = rnn.rnn(self._enc_cell, inputs, dtype=tf.float32)
      self.z_mean = tf.add(tf.matmul(enc_state, z_mean_w), z_mean_b)
      self.z_log_var = tf.add(tf.matmul(enc_state, z_logvar_w), z_logvar_b)
      #reparameterisaton trick
      eps = tf.random_normal((self.batch_size, self.n_latent), 0, 1, dtype=tf.float32)
      self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))
      
    with tf.variable_scope('decoder') as scope:
      dec_in_w = tf.Variable(self.initializer([self.n_latent, self._dec_cell.state_size],
                                              dtype=tf.float32))
      dec_in_b = tf.Variable(tf.zeros([self._dec_cell.state_size], dtype=tf.float32))
      dec_out_w = tf.Variable(self.initializer([self.n_hidden, self.elem_num], dtype=tf.float32))
      dec_out_b = tf.Variable(tf.zeros([self.elem_num], dtype=tf.float32))

      initial_dec_state = self.transfer_func(tf.add(tf.matmul(self.z, dec_in_w), dec_in_b))
      #initial_dec_state = enc_state # vanilla autoencoder
      
      if self.decode_with_input:
        dec_state = initial_dec_state
        dec_input = tf.zeros([self.batch_size, self.elem_num], dtype=tf.float32)
        dec_out = []
        for step in range(self.step_num):
          if step > 0: 
            scope.reuse_variables()
          dec_input, dec_state = self._dec_cell(dec_input, dec_state)
          dec_input = tf.nn.sigmoid(tf.matmul(dec_input, dec_out_w) + dec_out_b)
          dec_out.append(dec_input)
        if self.init_reverse:
          dec_out = dec_out[::-1]
        self.output = tf.transpose(tf.pack(dec_out), [1, 0, 2])
      else:
        dec_inputs = [tf.zeros([self.batch_size, self.elem_num], dtype=tf.float32)
                      for _ in range(self.step_num)]
        dec_out, dec_state = rnn.rnn(self._dec_cell, dec_inputs, 
                                     initial_state=initial_dec_state, dtype=tf.float32)
        if self.init_reverse:
          dec_out = dec_out[::-1]
        dec_output = tf.transpose(tf.pack(dec_out), [1, 0, 2])
        dec_out_w = tf.tile(tf.expand_dims(dec_out_w, 0), [self.batch_size, 1, 1])
        self.output = tf.nn.sigmoid(tf.batch_matmul(dec_output, dec_out_w) + dec_out_b)

    self.inp = tf.transpose(tf.pack(inputs), [1, 0, 2])
    #VAE loss
    inp = tf.reshape(self.inp, [self.batch_size, self.elem_num * self.step_num])
    out = tf.reshape(self.output, [self.batch_size, self.elem_num * self.step_num])
    encoder_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_var), 1)
    decoder_loss = -tf.reduce_sum(inp * tf.log(1e-10 + out) 
                                  + (1 - inp) * tf.log(1e-10 + (1 - out)), 1)
    self.loss = tf.reduce_mean(encoder_loss + decoder_loss)

    #self.loss = tf.reduce_mean(tf.square(self.inp - self.output)) # vanilla autoencoder
    self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

  def generate(self, sess, z_mu=None):
    if z_mu is None:
      z_mu = np.random.normal(size=(self.batch_size, self.n_latent))
    return sess.run(self.output, {self.z: z_mu})

def plot(x_sample, x_reconstruct):
    reconstruct_count = 5
    fig, ax = plt.subplots(2, reconstruct_count, figsize=(1.2*reconstruct_count, 3))
    for index in range(0, reconstruct_count):
      ax[0, index].imshow(np.reshape(x_sample[index], (28, 28)), vmin=0, vmax=1)
      ax[0, index].axis('off')
      ax[1, index].imshow(np.reshape(x_reconstruct[index], (28, 28)), vmin=0, vmax=1)
      ax[1, index].axis('off')
    plt.show()

def save_model(sess, saver):
  if FLAGS.save_dir:
    saver.save(sess, FLAGS.save_dir + '/model.ckpt')

def main(_):
  batch_size = 20
  n_hidden = 50
  n_latent = 20
  step_num = 28
  elem_num = 28
  iteration = 2000
  print_iter = 100
  x = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])
  x_list = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, x)]

  cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
  vae = LSTM_VAE(n_hidden, x_list, batch_size, elem_num, step_num, cell=cell, n_latent=n_latent)
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
    else:
      sess.run(tf.initialize_all_variables())
    for i in range(iteration):
      batch_xs, _ = mnist.train.next_batch(batch_size)
      batch_xs = batch_xs.reshape((batch_size, step_num, elem_num))
      sess.run(vae.train, {x: batch_xs})
      if (i % print_iter == 0):
        loss_val = sess.run(vae.loss, {x: batch_xs})
        print "iter %d:" % (i+1), loss_val
    sample_xs, _ = mnist.test.next_batch(batch_size)
    sample_xs = sample_xs.reshape((batch_size, step_num, elem_num))
    reconstruct_xs = sess.run(vae.output, {x: sample_xs})
    plot(sample_xs, reconstruct_xs)
    plot(vae.generate(sess), vae.generate(sess))
    save_model(sess, saver)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory for saving model')
  parser.add_argument('--load_model', type=str, default=None,
                      help='Directory from which to load model')
  parser.add_argument('--save_predictions', type=str, default=None,
                      help='Location to pickle predictions')
  FLAGS = parser.parse_args()
  tf.set_random_seed(1234)
  np.random.seed(1234)
  tf.app.run()


