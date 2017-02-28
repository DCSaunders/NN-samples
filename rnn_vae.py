from __future__ import division
import argparse
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn
FLAGS = None

class VAE(object):
  def __init__(self, batch_size, step_num, n_hidden, vocab_size, cell, n_latent, 
               initializer=tf.truncated_normal, transfer_func=tf.nn.softplus, annealing=False):
    self.batch_size = batch_size
    self.step_num = step_num 
    self.vocab_size = vocab_size
    self.n_hidden = n_hidden
    self.n_latent = n_latent
    self.kl_weight = 0 if annealing else 1
    self.z_gen = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_latent], name="zgen")
    self.enc_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder{}".format(i))
                       for i in range(self.step_num)]
    self.dec_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder{}".format(i))
                       for i in range(self.step_num + 1)]
    self.loss_weights = [tf.placeholder(tf.float32, shape=[None], name="weight{}".format(i))
                       for i in range(self.step_num + 1)]
    self.targets = [self.dec_inputs[i + 1] for i in range(len(self.dec_inputs) - 1)]
    self.targets.append(tf.placeholder(tf.int32, shape=[None], name="last_target"))
    self.initializer = initializer
    self.transfer_func = transfer_func
    self._enc_cell = cell
    self._dec_cell = cell
    self.build_network()

  def get_latent(self, enc_state):
    z_mean_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent],
                                            dtype=tf.float32))
    z_mean_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))
    z_logvar_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent],
                                              dtype=tf.float32))
    z_logvar_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))

    self.z_mean = tf.add(tf.matmul(enc_state, z_mean_w), z_mean_b)
    self.z_log_var = tf.add(tf.matmul(enc_state, z_logvar_w), z_logvar_b)
    eps = tf.random_normal((self.batch_size, self.n_latent), 0, 1, dtype=tf.float32)
    self.z = tf.add(self.z_mean, tf.mul(tf.sqrt(tf.exp(self.z_log_var)), eps))

  def build_network(self):
    with tf.variable_scope('encoder'):
      embedding = tf.Variable(self.initializer([self.vocab_size, self.n_hidden], dtype=tf.float32))
      embedded_inputs = [tf.nn.embedding_lookup(embedding, enc_input) 
                         for enc_input in self.enc_inputs]
      _, enc_state = tf.nn.rnn(self._enc_cell, embedded_inputs, dtype=tf.float32)

    self.get_latent(enc_state)
    dec_in_w = tf.Variable(self.initializer([self.n_latent, self._dec_cell.state_size],
                                              dtype=tf.float32))
    dec_in_b = tf.Variable(tf.zeros([self._dec_cell.state_size], dtype=tf.float32))
    dec_initial_state = self.transfer_func(tf.add(tf.matmul(self.z, dec_in_w), dec_in_b))
    dec_gen_initial_state = self.transfer_func(tf.add(tf.matmul(self.z_gen, dec_in_w), dec_in_b))

    with tf.variable_scope('decoder') as scope:
      softmax_w = tf.Variable(self.initializer([self.n_hidden, self.vocab_size]))
      softmax_b = tf.Variable(tf.zeros([self.vocab_size]))
      output_projection=[softmax_w, softmax_b]
      dec_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.dec_inputs, dec_initial_state,
                                                        self._dec_cell, self.vocab_size,
                                                        self.n_hidden, output_projection,
                                                        feed_previous=False)
      scope.reuse_variables()
      dec_test_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.dec_inputs, dec_initial_state,
                                                            self._dec_cell, self.vocab_size,
                                                            self.n_hidden, output_projection,
                                                            feed_previous=True,
                                                             update_embedding_for_previous=False)

      dec_gen_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(self.dec_inputs, dec_gen_initial_state,
                                                             self._dec_cell, self.vocab_size,
                                                             self.n_hidden, output_projection,
                                                             feed_previous=True,
                                                             update_embedding_for_previous=False)

      self.logits = [tf.add(tf.matmul(dec_out, softmax_w), softmax_b) 
                     for dec_out in dec_outs]
      self.output = [tf.argmax(tf.nn.softmax(logit), 1) for logit in self.logits]
      
      dec_gen_out = tf.reshape(tf.concat(1, dec_gen_outs), [-1, self.n_hidden])
      dec_test_out = tf.reshape(tf.concat(1, dec_test_outs), [-1, self.n_hidden])
      self.test_output = tf.reshape(
        tf.argmax(tf.nn.softmax(tf.matmul(dec_test_out, softmax_w) + softmax_b), 1),
        [self.batch_size, self.step_num + 1])
      self.gen_output = tf.reshape(
        tf.argmax(tf.nn.softmax(tf.matmul(dec_gen_out, softmax_w) + softmax_b), 1),
        [self.batch_size, self.step_num + 1])
    
    seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
        self.logits,
        self.targets,
        self.loss_weights)
    self.xentropy_loss  = tf.reduce_sum(seq_loss) / self.batch_size
    self.kl_loss = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                        - tf.square(self.z_mean)
                                        - tf.exp(self.z_log_var), 1)
    self.loss = tf.reduce_mean(self.kl_weight * self.kl_loss + self.xentropy_loss)
    self.train = tf.train.AdamOptimizer().minimize(self.loss)

  def generate(self, sess, GO_ID, z_mu=None):
    if z_mu is None:
      z_mu = np.random.normal(size=(self.batch_size, self.n_latent))
    input_feed = {self.z_gen: z_mu}
    input_feed[self.dec_inputs[0].name] = GO_ID * np.ones(self.batch_size)
    return sess.run(self.gen_output, input_feed)
    
def get_batch(batch_size, step_num, max_val, vae, GO_ID):
  GO_ID = 0
  input_feed = {}
  r = np.random.randint(low=1, high=max_val, size=batch_size).reshape([batch_size, 1])
  r = np.tile(r, (1, step_num))
  d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num])
  d = np.tile(d, (batch_size, 1))
  enc_input_data = r + d
  dec_input_data = np.array([[GO_ID] + list(seq) for seq in enc_input_data])
  target_data = [dec_in[1:] for dec_in in vae.dec_inputs]
  for l in range(step_num):
    input_feed[vae.enc_inputs[l].name] = enc_input_data[:, l]
  for l in range(step_num + 1):
    input_feed[vae.dec_inputs[l].name] = dec_input_data[:, l]
    input_feed[vae.loss_weights[l].name] = np.ones(batch_size)
  input_feed[vae.targets[-1].name] = np.zeros(batch_size)
  return input_feed, enc_input_data


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

def main(_):
  batch_size = 50
  n_hidden = 30
  n_latent = 8
  step_num = 8
  iteration = 300
  print_iter = 100
  max_val = 25
  GO_ID = 1
  vocab_size = max_val + step_num
  kl_weight_rate = 1/10

  cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
  vae = VAE(batch_size, step_num, n_hidden, vocab_size, cell, n_latent, annealing=FLAGS.annealing)
  loss_hist = {'kl': [], 'xentropy': []}
  saver = tf.train.Saver()

  with tf.Session() as sess:
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
    else:
      sess.run(tf.initialize_all_variables())

    for i in range(iteration):
      if FLAGS.annealing:
        vae.kl_weight = 1 - np.exp(-i * kl_weight_rate)
      input_feed, _ = get_batch(batch_size, step_num, max_val, vae, GO_ID)

      if FLAGS.plot_loss:
        _, kl_loss, xentropy_loss = sess.run([vae.train, vae.kl_loss, vae.xentropy_loss], input_feed)
        loss_hist['kl'].append(np.mean(kl_loss))
        loss_hist['xentropy'].append(np.mean(xentropy_loss))
      else:
        sess.run(vae.train, input_feed)
      
      if (i % print_iter == 0):
        loss_val = sess.run(vae.loss, input_feed)
        print "iter {}, train loss {}".format((i+1), loss_val)
        
    if FLAGS.plot_loss:
      plot_loss(loss_hist['kl'], loss_hist['xentropy'])

    input_feed, input_data = get_batch(batch_size, step_num, max_val, vae, GO_ID)
    output = sess.run(vae.test_output, input_feed)
    for in_, out_ in zip(input_data, output):
      print "Input: {}, Output: {}".format(in_, out_[:-1])
    '''
    print "Generating"
    gen_out = vae.generate(sess, GO_ID)
    for out in gen_out:
      print out[:-1]
    '''
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
