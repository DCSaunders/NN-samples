import argparse
import numpy as np
import tensorflow as tf
FLAGS = None

class LSTMAutoencoder(object):
  def __init__(self, batch_size, step_num, n_hidden, vocab_size, cell, reverse=True):
    self.batch_size = batch_size
    self.step_num = step_num 
    self.vocab_size = vocab_size
    self.n_hidden = n_hidden
    self.inputs = tf.placeholder(tf.int32, [batch_size, step_num])
    self.targets = tf.placeholder(tf.int32, [batch_size, step_num])
    self._enc_cell = cell
    self._dec_cell = cell

    with tf.variable_scope('encoder'):
      embedding = tf.get_variable("embedding", [self.vocab_size, self.n_hidden])
      embedded_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
      enc_inputs = [tf.squeeze(input_, [1])
                    for input_ in tf.split(1, self.step_num, embedded_inputs)]
      _, self.enc_state = tf.nn.rnn(self._enc_cell, enc_inputs, dtype=tf.float32)

    with tf.variable_scope('decoder') as scope:
      softmax_w = tf.get_variable("softmax_w", [self.n_hidden, self.vocab_size])
      softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
      output_projection=(softmax_w, softmax_b)
      dec_initial_state = self.enc_state
      dec_inputs = [tf.squeeze(input_, [1])
                    for input_ in tf.split(1, self.step_num, self.inputs)]

      dec_outs, _ = tf.nn.seq2seq.embedding_rnn_decoder(dec_inputs, dec_initial_state,
                                                        self._dec_cell, vocab_size,
                                                        n_hidden, output_projection,
                                                        feed_previous=False)
      dec_out = tf.reshape(tf.concat(1, dec_outs), [-1, self.n_hidden])
      self.logits = tf.matmul(dec_out, softmax_w) + softmax_b
      self.output = tf.argmax(tf.nn.softmax(self.logits), 1)
      
    seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
        [self.logits],
        [tf.reshape(self.targets, [-1])],
        [tf.ones([self.batch_size * self.step_num])])
    self.loss  = tf.reduce_sum(seq_loss) / self.batch_size
    self.train = tf.train.AdamOptimizer().minimize(self.loss)

def get_batch(batch_size, step_num, max_val):
  r = np.random.randint(max_val, size=batch_size).reshape([batch_size, 1])
  r = np.tile(r, (1, step_num))
  d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num])
  d = np.tile(d, (batch_size, 1))
  inputs = r + d
  targets = inputs
  return inputs, targets

def main(_):
  batch_size = 50
  n_hidden = 12
  step_num = 8
  iteration = 500
  print_iter = 100
  max_val = 20
  vocab_size = max_val + step_num + 1 # randint can generate 0

  cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
  ae = LSTMAutoencoder(batch_size, step_num, n_hidden, vocab_size, cell=cell)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for i in range(iteration):
      inputs, targets = get_batch(batch_size, step_num, max_val)
      sess.run(ae.train, {ae.inputs: inputs, ae.targets: targets})
      
      if (i % print_iter == 0):
        loss_val = sess.run(ae.loss, {ae.inputs: inputs, ae.targets: targets})
        print "iter {}, train loss {}".format((i+1), loss_val)

    inputs, targets = get_batch(batch_size, step_num, max_val)
    output = sess.run(ae.output, {ae.inputs: inputs, ae.targets: targets})
    output = output.reshape((batch_size, step_num))
    for in_, out_ in zip(inputs, output):
      print "Input: {}, Output: {}".format(in_, out_)

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
