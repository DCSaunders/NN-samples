from __future__ import division
import argparse
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None


class LSTM_VAE(object):
  def __init__(self, n_hidden, inputs, batch_size, elem_num, step_num, cell,
               learning_rate=0.001, reverse=True, n_latent=10, transfer_func=tf.nn.softplus, initializer=tf.truncated_normal, annealing=False):
    """
    n_hidden : number of hidden elements of each LSTM unit.
    inputs : a list of input tensors with size (batch_size x elem_num)
    cell : an rnn cell object
    learning_rate: learning rate for Adam optimizer
    reverse : Option to decode in reverse order.
    decode_without_input : Option to decode without input.
    n_latent: number of latent states
    transfer_func: function on latent state to give decoder initial state
    initializer: tf initializer function for weights
    """
    self.batch_size = batch_size
    self.kl_weight = 0 if annealing else 1
    self.elem_num = elem_num
    self.step_num = step_num
    self.n_hidden = n_hidden
    self.n_latent = n_latent
    self._enc_cell = cell
    self._dec_cell = cell
    self.initializer = initializer
    self.transfer_func = transfer_func
    self.learning_rate = learning_rate
    self.reverse = reverse
    self.inputs = inputs
    self.build_network()

  def build_network(self):
    with tf.variable_scope('encoder'):
      z_mean_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent]))
      z_mean_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))
      z_logvar_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent]))
      z_logvar_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))

      _, enc_state = rnn.rnn(self._enc_cell, self.inputs, dtype=tf.float32)
      self.z_mean = tf.add(tf.matmul(enc_state, z_mean_w), z_mean_b)
      self.z_log_var = tf.add(tf.matmul(enc_state, z_logvar_w), z_logvar_b)
      eps = tf.random_normal((self.batch_size, self.n_latent), 0, 1, dtype=tf.float32)
      self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))
      
    with tf.variable_scope('decoder') as scope:
      dec_in_w = tf.Variable(self.initializer([self.n_latent, self._dec_cell.state_size],
                                              dtype=tf.float32))
      dec_in_b = tf.Variable(tf.zeros([self._dec_cell.state_size], dtype=tf.float32))
      dec_out_w = tf.Variable(self.initializer([self.n_hidden, self.elem_num], dtype=tf.float32))
      dec_out_b = tf.Variable(tf.zeros([self.elem_num], dtype=tf.float32))

      initial_dec_state = self.transfer_func(tf.add(tf.matmul(self.z, dec_in_w), dec_in_b))
      dec_out, _ = seq2seq.rnn_decoder(self.inputs, initial_dec_state, self._dec_cell)
      if self.reverse:
        dec_out = dec_out[::-1]
      dec_output = tf.transpose(tf.pack(dec_out), [1, 0, 2])
      batch_dec_out_w = tf.tile(tf.expand_dims(dec_out_w, 0), [self.batch_size, 1, 1])
      self.output = tf.nn.sigmoid(tf.batch_matmul(dec_output, batch_dec_out_w) + dec_out_b)

      scope.reuse_variables()
      dec_gen_input = [0.5 * tf.ones([self.batch_size, self.elem_num],
                                dtype=tf.float32) for _ in range(self.step_num)]
      self.z_gen = tf.placeholder(tf.float32, [self.batch_size, self.n_latent])
      dec_gen_state = self.transfer_func(
        tf.add(tf.matmul(self.z_gen, dec_in_w), dec_in_b))
      dec_gen_out, _ = seq2seq.rnn_decoder(
        dec_gen_input, dec_gen_state, self._dec_cell)
      if self.reverse:
        dec_gen_out = dec_gen_out[::-1]
      dec_gen_output = tf.transpose(tf.pack(dec_gen_out), [1, 0, 2]) 
      self.gen_output = tf.nn.sigmoid(tf.batch_matmul(dec_gen_output, batch_dec_out_w) + dec_out_b)
    
    self.inp = tf.transpose(tf.pack(self.inputs), [1, 0, 2])
    self.train_loss = self.get_loss()
    self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.train_loss)

  def get_loss(self):
    out = tf.reshape(self.output, [self.batch_size, self.elem_num * self.step_num])
    inp = tf.reshape(self.inp, [self.batch_size, self.elem_num * self.step_num])
    self.kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                        - tf.square(self.z_mean)
                                        - tf.exp(self.z_log_var), 1)
    self.xentropy_loss = -tf.reduce_sum(inp * tf.log(1e-10 + out) 
                                  + (1 - inp) * tf.log(1e-10 + (1 - out)), 1)
    return tf.reduce_mean(self.kl_weight * self.kl_loss + self.xentropy_loss)
      

  def generate(self, x, sess, z_mu=None):
    if z_mu is None:
      z_mu = np.random.normal(size=(self.batch_size, self.n_latent))
    return sess.run(self.gen_output, {self.z_gen: z_mu})

def plot(x_sample, x_reconstruct, name='plot_rnn_vae'):
  reconstruct_count = 5
  fig, ax = plt.subplots(2, reconstruct_count, figsize=(1.2*reconstruct_count, 3))
  for index in range(0, reconstruct_count):
    ax[0, index].imshow(np.reshape(x_sample[index], (28, 28)), vmin=0, vmax=1)
    ax[0, index].axis('off')
    ax[1, index].imshow(np.reshape(x_reconstruct[index], (28, 28)), vmin=0, vmax=1)
    ax[1, index].axis('off')
  if FLAGS.save_fig:
    plt.savefig('{}/{}.png'.format(FLAGS.save_fig, name), bbox_inches='tight')
  else:
    plt.show()

def plot_loss(kl_loss, xentropy):
  plt.clf()
  x = np.array(range(len(kl_loss)))
  plt.plot(x, kl_loss, 'r', label='KL loss')
  plt.plot(x, xentropy, 'b', label='Cross-entropy')
  plt.legend()
  if FLAGS.save_fig:
    plt.savefig('{}/loss.png'.format(FLAGS.save_fig), bbox_inches='tight')
  else:
    plt.show()

def save_model(sess, saver):
  if FLAGS.save_dir:
    saver.save(sess, FLAGS.save_dir + '/model.ckpt')

def dropout(x):
  p = FLAGS.decoder_dropout

def main(_):
  batch_size = 50
  n_hidden = 50
  n_latent = 20
  step_num = 28
  elem_num = 28
  iteration = 1100
  print_iter = 100
  kl_weight_rate = 1 / 100

  x = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])
  x_list = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, x)]
  loss_hist = {'kl': [], 'xentropy': []}

  cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
  vae = LSTM_VAE(n_hidden, x_list, batch_size, elem_num, step_num, cell=cell, n_latent=n_latent, annealing=FLAGS.annealing)
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
    else:
      sess.run(tf.initialize_all_variables())
    for i in range(iteration):
      if FLAGS.annealing:
        vae.kl_weight = 1 - np.exp(-i * kl_weight_rate)
      batch_xs, _ = mnist.train.next_batch(batch_size)
      dropout(batch_xs)
      batch_xs = batch_xs.reshape((batch_size, step_num, elem_num))
      if FLAGS.plot_loss:
        _, kl_loss, xentropy_loss = sess.run([vae.train, vae.kl_loss, vae.xentropy_loss],
                                         {x: batch_xs})
        loss_hist['kl'].append(np.mean(kl_loss))
        loss_hist['xentropy'].append(np.mean(xentropy_loss))
      else:
        sess.run(vae.train, {x: batch_xs})

      if (i % print_iter == 0):
        batch_xs, _ = mnist.validation.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, step_num, elem_num))
        train_loss = sess.run(vae.train_loss, {x: batch_xs})
        print "iter {}, train loss {}".format((i+1), train_loss)
    sample_xs, _ = mnist.test.next_batch(batch_size)
    sample_xs = sample_xs.reshape((batch_size, step_num, elem_num))
    reconstruct_xs = sess.run(vae.output, {x: sample_xs})
    plot(sample_xs, reconstruct_xs, 'reconstruct')
    plot(vae.generate(x, sess), vae.generate(x, sess), 'generate')
    if FLAGS.plot_loss:
      plot_loss(loss_hist['kl'], loss_hist['xentropy'])
    save_model(sess, saver)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory for saving model')
  parser.add_argument('--load_model', type=str, default=None,
                      help='Directory from which to load model')
  parser.add_argument('--save_fig', type=str, default=None,
                      help='Location to save plots')
  parser.add_argument('--plot_loss', default=False, action='store_true',
                      help='Set if plotting encoder/decoder loss over time')
  parser.add_argument('--annealing', default=False, action='store_true',
                      help='Set if initially not weighting KL loss')
  parser.add_argument('--decoder_dropout', type=float, default=1.0, 
                      help='Proportion of input to replace with dummy value on decoder input')


  FLAGS = parser.parse_args()
  tf.set_random_seed(1234)
  np.random.seed(1234)
  tf.app.run()


