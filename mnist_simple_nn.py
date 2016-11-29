from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  n_in = 784
  n_out = 10
  n_hidden = 200
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  w_in = tf.Variable(tf.random_normal([n_in, n_hidden]))
  b_in = tf.Variable(tf.random_normal([n_hidden]))
  w_out = tf.Variable(tf.random_normal([n_hidden, n_out]))
  b_out = tf.Variable(tf.random_normal([n_out]))
  # Create the model
  x = tf.placeholder(tf.float32, [None, n_in])
  h = tf.nn.relu(tf.add(tf.matmul(x, w_in), b_in))
  y = tf.add(tf.matmul(h, w_out), b_out)

  batch_size = 100
  labels = tf.placeholder(tf.float32, [None, n_out])
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, labels))
  optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
  with tf.Session() as sess:
    # Train
    sess.run(tf.initialize_all_variables())
    for _ in range(5000):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      sess.run(optimizer, feed_dict={x: batch_xs, labels: batch_ys})
      #print(sess.run(tf.nn.softmax(y), feed_dict={x: batch_xs}))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        labels: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()

