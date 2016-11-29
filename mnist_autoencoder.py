from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
FLAGS = None


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Parameters
  learning_rate = 0.01
  training_epochs = 20
  batch_size = 100
  batch_count = int(mnist.train.num_examples / batch_size)
  display_step = 1
  reconstruct_count = 10
  
  im_size = 28
  n_in = 784
  n_hidden_1 = 200
  n_hidden_2 = 100
  x = tf.placeholder("float", [None, n_in])
  enc_w_1 = tf.Variable(tf.random_normal([n_in, n_hidden_1]))
  enc_b_1 = tf.Variable(tf.random_normal([n_hidden_1]))
  enc_w_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
  enc_b_2 = tf.Variable(tf.random_normal([n_hidden_2]))

  dec_w_1 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))
  dec_b_1 = tf.Variable(tf.random_normal([n_hidden_1]))
  dec_w_2 = tf.Variable(tf.random_normal([n_hidden_1, n_in]))
  dec_b_2 = tf.Variable(tf.random_normal([n_in]))
  
  enc_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, enc_w_1), enc_b_1))
  enc_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(enc_layer_1, enc_w_2), enc_b_2))

  dec_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(enc_layer_2, dec_w_1), dec_b_1))
  dec_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(dec_layer_1, dec_w_2), dec_b_2))
  
  y_pred = dec_layer_2
  y_true = x
  cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
  optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

  with tf.Session() as sess:
    tf.set_random_seed(1234)
    sess.run(tf.initialize_all_variables())
    # Train
    for epoch in range(training_epochs):
      avg_cost = 0
      for batch in range(batch_count):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        _, batch_cost = sess.run([optimizer, cost], feed_dict={x: batch_xs})
        avg_cost += batch_cost / batch_count
      if epoch % display_step == 0:
        print("Epoch: {:03d}, cost={:.9f}".format(epoch + 1, avg_cost))
    
    # Dump hidden layer vectors
    with open(FLAGS.data_dir + '/hidden_train', 'w') as f:
      for im in mnist.train.images:
        f.write(sess.run(enc_layer_1, feed_dict={x: [im]}) + '\n')
    with open(FLAGS.data_dir + '/hidden_test', 'w') as f:
      for im in mnist.train.images:
        f.write(sess.run(enc_layer_1, feed_dict={x: [im]}) + '\n')

    # Reconstruct some examples
    reconstructions = sess.run(y_pred, feed_dict={
      x: mnist.test.images[:reconstruct_count]})
    fig, ax = plt.subplots(2, reconstruct_count, figsize=(reconstruct_count, 2))
    for im in range(reconstruct_count):
      ax[0][im].imshow(np.reshape(mnist.test.images[im], (im_size, im_size)))
      ax[1][im].imshow(np.reshape(reconstructions[im], (im_size, im_size)))
    fig.show()
    fig.savefig('mnist_autoenc_reconstr.png')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  np.random.seed(1234)
  tf.app.run()

