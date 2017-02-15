# Basic libraries
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
tf.set_random_seed(1234)
np.random.seed(1234)


class LSTMAutoencoder(object):
  """Basic version of LSTM-autoencoder.
  (cf. http://arxiv.org/abs/1502.04681)
  Usage:
    ae = LSTMAutoencoder(n_hidden, inputs)
    sess.run(ae.train)
  """

  def __init__(self, n_hidden, inputs, cell,
               optimizer=None, reverse=True, 
               decode_without_input=False, n_latent=10):
    """
    Args:
      n_hidden : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size (batch_size x elem_num)
      cell : an rnn cell object
      optimizer : optimizer for rnn (defaults to Adam)
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
      n_latent: number of latent states
    """

    self.batch_size = inputs[0].get_shape().as_list()[0]
    self.elem_num = inputs[0].get_shape().as_list()[1]
    self._enc_cell = cell
    self._dec_cell = cell

    with tf.variable_scope('encoder'):
      _, enc_state = rnn.rnn(self._enc_cell, inputs, dtype=tf.float32)

    with tf.variable_scope('decoder') as scope:
      dec_weight = tf.Variable(
        tf.truncated_normal([n_hidden, self.elem_num], dtype=tf.float32))
      dec_bias = tf.Variable(
        tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32))

      if decode_without_input:
        dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                      for _ in range(len(inputs))]
        dec_outputs, dec_state = rnn.rnn(self._dec_cell, dec_inputs, 
                                         initial_state=enc_state, dtype=tf.float32)
        """dec_output : (step_num x n_hidden)
          dec_weight : (n_hidden x elem_num)
          dec_bias : (elem_num)
          output : (step_num x elem_num)
          input : (step_num x elem_num)
        """
        if reverse:
          dec_outputs = dec_outputs[::-1]
        dec_output = tf.transpose(tf.pack(dec_outputs), [1, 0, 2])
        dec_weight = tf.tile(tf.expand_dims(dec_weight, 0), [self.batch_size, 1, 1])
        self.output = tf.batch_matmul(dec_output, dec_weight) + dec_bias

      else: 
        dec_state = enc_state
        dec_input = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_outputs = []
        for step in range(len(inputs)):
          if step > 0: 
            scope.reuse_variables()
          dec_input, dec_state = self._dec_cell(dec_input, dec_state)
          dec_input_ = tf.matmul(dec_input, dec_weight) + dec_bias
          dec_outputs.append(dec_input)
        if reverse:
          dec_outputs = dec_outputs[::-1]
        self.output = tf.transpose(tf.pack(dec_outputs), [1, 0, 2])

    self.inp = tf.transpose(tf.pack(inputs), [1, 0, 2])
    self.loss = tf.reduce_mean(tf.square(self.inp - self.output))

    if optimizer is None :
      self.train = tf.train.AdamOptimizer().minimize(self.loss)
    else:
      self.train = optimizer.minimize(self.loss)

batch_size = 50
n_hidden = 20
n_latent = 10
step_num = 8
elem_num = 1
iteration = 500
print_iter = 100
max_val = 20
p_input = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])
p_input_list = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True)
ae = LSTMAutoencoder(n_hidden, p_input_list, cell=cell, decode_without_input=True, n_latent=n_latent)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for i in range(iteration):
    """Every sequence has size batch_size * step_num * elem_num 
      Each sequence is a list of integers increasing by 1.
      Initial value in each sequence is in range 0:max_value.
      (ex. [8. 9. 10. 11. 12. 13. 14. 15])
    """
    # Set first value for each sequence in batch
    r = np.random.randint(max_val, size=batch_size).reshape([batch_size, 1, 1])
    r = np.tile(r, (1, step_num, elem_num))
    # Set up linearly increasing sequence of same length
    d = np.linspace(0, step_num, step_num, endpoint=False).reshape(
      [1, step_num, elem_num])
    d = np.tile(d, (batch_size, 1, 1))
    random_sequences = r+d

    sess.run(ae.train, {p_input: random_sequences})
    if (i % print_iter == 0):
        loss_val = sess.run(ae.loss, {p_input: random_sequences})
        print "iter %d:" % (i+1), loss_val

  inp, output =  sess.run([ae.inp, ae.output], {p_input:r+d})
  print "train result: "
  print "input: ", inp[0,:,:].flatten()
  print "output: ", output[0,:,:].flatten()
