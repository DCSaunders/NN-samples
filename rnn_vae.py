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

  def __init__(self, n_hidden, inputs, cell, learning_rate=0.001, reverse=True, 
               decode_without_input=False, n_latent=10, transfer_func=tf.nn.softplus):
    """
    Args:
      n_hidden : number of hidden elements of each LSTM unit.
      inputs : a list of input tensors with size (batch_size x elem_num)
      cell : an rnn cell object
      learning_rate: learning rate for Adam optimizer
      reverse : Option to decode in reverse order.
      decode_without_input : Option to decode without input.
      n_latent: number of latent states
    """

    self.batch_size = inputs[0].get_shape().as_list()[0]
    self.elem_num = inputs[0].get_shape().as_list()[1]
    self.n_latent = n_latent
    self._enc_cell = cell
    self._dec_cell = cell
    self.initializer = tf.truncated_normal
    self.transfer_func = transfer_func
    self.learning_rate = learning_rate

    with tf.variable_scope('encoder'):
      z_mean_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent]))
      z_mean_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))
      z_logvar_w = tf.Variable(self.initializer([self._enc_cell.state_size, self.n_latent]))
      z_logvar_b = tf.Variable(tf.zeros([self.n_latent], dtype=tf.float32))

      _, enc_state = rnn.rnn(self._enc_cell, inputs, dtype=tf.float32)
      z_mean = tf.add(tf.matmul(enc_state, z_mean_w), z_mean_b)
      z_log_var = tf.add(tf.matmul(enc_state, z_logvar_w), z_logvar_b)
      #reparameterisaton trick
      eps = tf.random_normal((self.batch_size, self.n_latent), 0, 1, dtype=tf.float32)
      self.z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_var)), eps))
      
    with tf.variable_scope('decoder') as scope:
      dec_in_w = tf.Variable(self.initializer([self.n_latent, self._dec_cell.state_size],
                                              dtype=tf.float32))
      dec_in_b = tf.Variable(tf.zeros([self._dec_cell.state_size], dtype=tf.float32))
      dec_out_w = tf.Variable(self.initializer([n_hidden, self.elem_num], dtype=tf.float32))
      dec_out_b = tf.Variable(tf.zeros([self.elem_num], dtype=tf.float32))

      initial_dec_state = self.transfer_func(tf.add(tf.matmul(self.z, dec_in_w), dec_in_b))
      #initial_dec_state = enc_state # vanilla autoencoder

      if decode_without_input:
        dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                      for _ in range(len(inputs))]
        dec_out, dec_state = rnn.rnn(self._dec_cell, dec_inputs, 
                                     initial_state=initial_dec_state, dtype=tf.float32)
        """dec_out : (step_num x n_hidden)
          dec_out_w : (n_hidden x elem_num)
          dec_out_b : (elem_num)
          output : (step_num x elem_num)
          inp : (step_num x elem_num)
        """
        if reverse:
          dec_out = dec_out[::-1]
        dec_output = tf.transpose(tf.pack(dec_out), [1, 0, 2])
        dec_out_w = tf.tile(tf.expand_dims(dec_out_w, 0), [self.batch_size, 1, 1])
        self.output = tf.batch_matmul(dec_output, dec_out_w) + dec_out_b

      else: 
        dec_state = initial_dec_state
        dec_input = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
        dec_out = []
        for step in range(len(inputs)):
          if step > 0: 
            scope.reuse_variables()
          dec_input, dec_state = self._dec_cell(dec_input, dec_state)
          dec_input = tf.matmul(dec_input, dec_out_w) + dec_out_b
          dec_out.append(dec_input)
        if reverse:
          dec_out = dec_out[::-1]
        self.output = tf.transpose(tf.pack(dec_out), [1, 0, 2])

    self.inp = tf.transpose(tf.pack(inputs), [1, 0, 2])
    self.loss = tf.reduce_mean(tf.square(self.inp - self.output))
    self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

batch_size = 50
n_hidden = 30
n_latent = 15
step_num = 8
elem_num = 1
iteration = 500
print_iter = 100
max_val = 20
p_input = tf.placeholder(tf.float32, [batch_size, step_num, elem_num])
p_input_list = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]

cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
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
