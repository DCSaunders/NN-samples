from __future__ import division
import argparse
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import subprocess
from tensorflow.python.ops import rnn
FLAGS = None

class config(object):
  GO_ID = 1
  EOS_ID = 2
  dev_ref = 'dev_ref'
  dev_out = 'dev_out'

class VAE(object):
  def __init__(self, batch_size, max_len, n_hidden, vocab_size, cell, n_latent, 
               initializer=tf.truncated_normal, transfer_func=tf.nn.softplus, annealing=False):
    self.batch_size = batch_size
    self.max_len = max_len 
    self.vocab_size = vocab_size
    self.n_hidden = n_hidden
    self.n_latent = n_latent
    self.kl_weight = 0 if annealing else 1
    self.z_gen = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_latent], name="zgen")
    self.enc_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder{}".format(i))
                       for i in range(self.max_len)]
    self.dec_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder{}".format(i))
                       for i in range(self.max_len + 1)]
    self.loss_weights = [tf.placeholder(tf.float32, shape=[None], name="weight{}".format(i))
                       for i in range(self.max_len + 1)]
    self.seq_len = tf.placeholder(tf.int32, shape=[self.batch_size], name="seqlen")
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
      _, enc_state = tf.nn.rnn(self._enc_cell, embedded_inputs, dtype=tf.float32,
                               sequence_length=self.seq_len)

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
        [self.batch_size, self.max_len + 1])
      self.gen_output = tf.reshape(
        tf.argmax(tf.nn.softmax(tf.matmul(dec_gen_out, softmax_w) + softmax_b), 1),
        [self.batch_size, self.max_len + 1])
    
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

  def generate(self, sess, z_mu=None):
    if z_mu is None:
      z_mu = np.random.normal(size=(self.batch_size, self.n_latent))
    input_feed = {self.z_gen: z_mu}
    input_feed[self.dec_inputs[0].name] = config.GO_ID * np.ones(self.batch_size)
    return sess.run(self.gen_output, input_feed)
    


def get_batch(max_val, vae):
  input_feed = {}
  enc_input_data = []
  seq_len = []
  loss_weights = np.ones((vae.max_len + 1, vae.batch_size))
  for seq in range(vae.batch_size):
    rand_len = np.random.randint(low=2, high=vae.max_len)
    r = np.random.randint(low=4, high=max_val) * np.ones(rand_len) + range(rand_len)
    r[-1] = config.EOS_ID
    enc_input_data.append(np.pad(r, (0, vae.max_len - rand_len), 'constant'))
    seq_len.append(rand_len)
    loss_weights[rand_len:, seq] = 0

  enc_input_data = np.array(enc_input_data)
  dec_input_data = np.array([[config.GO_ID] + list(seq) for seq in enc_input_data])
  target_data = [dec_in[1:] for dec_in in vae.dec_inputs]

  for l in range(vae.max_len):
    input_feed[vae.enc_inputs[l].name] = enc_input_data[:, l]
  for l in range(vae.max_len + 1):
    input_feed[vae.dec_inputs[l].name] = dec_input_data[:, l]
    input_feed[vae.loss_weights[l].name] = loss_weights[l, :]
  input_feed[vae.targets[-1].name] = np.zeros(vae.batch_size)
  input_feed[vae.seq_len.name] = np.array(seq_len)
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

def save_model(sess, saver, name=''):
  if FLAGS.save_dir:
    fname = '{}/model.ckpt{}'.format(name)
    saver.save(sess, FLAGS.save_dir + '/model.ckpt')

def bleu_eval(vae, dev_feed, sess, saver, best_bleu):
  dev_out = sess.run(vae.test_output, dev_feed)
  save_batch(dev_out, config.dev_out)
  cat = subprocess.Popen(("cat", config.dev_out), stdout=subprocess.PIPE)
  try:
    multibleu = subprocess.check_output(("/home/mifs/ds636/code/scripts/multi-bleu.perl", 
                                         config.dev_ref), stdin=cat.stdout)
    print "{}".format(multibleu)
    m = re.match("BLEU = ([\d.]+),", multibleu)
    new_bleu = float(m.group(1))
    if new_bleu > best_bleu:
      print 'Model achieves new best bleu'
      save_model(sess, saver, name='-dev_bleu')
    return new_bleu
  except Exception, e:
    print "Multi-bleu error: {}".format(e)
    return 0.0

def get_dev_ref(vae, max_val):
  dev_feed, dev_data = get_batch(max_val, vae)
  save_batch(dev_data, config.dev_ref)
  return dev_feed

def save_batch(data, fname):
  with open(fname, 'w') as f_out:
    for out in data:
      out = out.astype(int).tolist()
      try:
        out = out[:out.index(config.EOS_ID)]
      except ValueError:
        pass
      f_out.write('{}\n'.format(' '.join(map(str, out))))    
   
def main(_):
  batch_size = 50
  n_hidden = 30
  n_latent = 10
  max_len = 15
  max_iter = 500
  dev_eval_iter = 100
  max_val = 25
  min_val = 4
  vocab_size = max_val + max_len
  kl_weight_rate = 1/10
  best_dev_bleu = 0.0
             
  cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
  vae = VAE(batch_size, max_len, n_hidden, vocab_size, cell, n_latent, annealing=FLAGS.annealing)
  loss_hist = {'kl': [], 'xentropy': []}
  saver = tf.train.Saver()
  dev_feed = get_dev_ref(vae, max_val)

  with tf.Session() as sess:
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
    else:
      sess.run(tf.initialize_all_variables())

    for i in range(max_iter):
      if FLAGS.annealing:
        vae.kl_weight = 1 - np.exp(-i * kl_weight_rate)
      input_feed, _ = get_batch(max_val, vae)

      if FLAGS.plot_loss:
        _, kl_loss, xentropy_loss = sess.run([vae.train, vae.kl_loss, vae.xentropy_loss], input_feed)
        loss_hist['kl'].append(np.mean(kl_loss))
        loss_hist['xentropy'].append(np.mean(xentropy_loss))
      else:
        sess.run(vae.train, input_feed)
      
      if (i % dev_eval_iter == 0):
        loss_val = sess.run(vae.loss, input_feed)
        dev_bleu = bleu_eval(vae, dev_feed, sess, saver, best_dev_bleu)
        best_dev_bleu = max(dev_bleu, best_dev_bleu)
        print "iter {}, train loss {}, dev BLEU {}".format((i+1), loss_val, dev_bleu)
        
    if FLAGS.plot_loss:
      plot_loss(loss_hist['kl'], loss_hist['xentropy'])

    input_feed, input_data = get_batch(max_val, vae)
    output = sess.run(vae.test_output, input_feed)
    for in_, out_ in zip(input_data, output):
      if config.EOS_ID in out_:
        out_ = out_[:list(out_).index(config.EOS_ID)]
      print "Input: {}, Output: {}".format(in_, out_)
    '''
    print "Generating"
    gen_out = vae.generate(sess)
    for out in gen_out:
      print out[:list(out).index(config.EOS_ID)]
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
