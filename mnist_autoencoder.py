from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import collections

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
FLAGS = None

def get_mnist():
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  return mnist

def get_samples():
  label_samples = collections.defaultdict(list)
  label_scores = collections.defaultdict(list)
  with open(FLAGS.load_samples, 'rb') as f_in:
    u = cPickle.Unpickler(f_in)
    while True:
      try:
        entry = u.load()
        label_samples[entry['label']].append(entry['states'])
        label_scores[entry['label']].append(entry['score'])
      except (EOFError):
        break
  return label_samples, label_scores

def main(_):
  # Parameters
  learning_rate = 0.01
  training_epochs = 20
  batch_size = 100
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
  saver = tf.train.Saver()

  with tf.Session() as sess:
    tf.set_random_seed(1234)
    if FLAGS.load_model:
      saver.restore(sess, FLAGS.load_model)
      if FLAGS.load_samples:
        label_sample_dict, label_score_dict = get_samples()
        print('Reconstructing {} from each component'.format(reconstruct_count))
        for label in label_sample_dict:
          if FLAGS.plot:
            if len(label_sample_dict[label]) < reconstruct_count:
              continue
            reconstructions = sess.run(y_pred, feed_dict={
              enc_layer_2: label_sample_dict[label][:reconstruct_count]})
            scores = label_score_dict[label][:reconstruct_count]
            fig, ax = plt.subplots(1, reconstruct_count, 
                                   figsize=(reconstruct_count, 2))
            fig.suptitle('Samples from GMM component {}'.format(label)) 
            for im in range(reconstruct_count):
              ax[im].set_title('{:.2f}'.format(scores[im]))
              ax[im].imshow(np.reshape(reconstructions[im], (im_size, im_size)))
            plt.show()
        reconstructions = []
        for label in label_sample_dict:
          reconstructions.extend(
            sess.run(y_pred, feed_dict={enc_layer_2: label_sample_dict[label]}))
        if FLAGS.save_reconstructions:
          with open(FLAGS.save_reconstructions, 'wb') as f_out:
            cPickle.dump(reconstructions, f_out)
    else:
      mnist = get_mnist()
      batch_count = int(mnist.train.num_examples / batch_size)
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
      reconstruct(sess, mnist, y_pred, x, reconstruct_count, im_size)
    if FLAGS.save_dir:
      save_model(sess, saver, mnist, enc_layer_2, x)

def reconstruct(sess, mnist, y_pred, x, reconstruct_count, im_size):
  # Reconstruct some examples
  reconstructions = sess.run(y_pred, feed_dict={
    x: mnist.test.images[:reconstruct_count]})
  fig, ax = plt.subplots(2, reconstruct_count, figsize=(reconstruct_count, 2))
  for im in range(reconstruct_count):
    ax[0][im].imshow(np.reshape(mnist.test.images[im], (im_size, im_size)))
    ax[1][im].imshow(np.reshape(reconstructions[im], (im_size, im_size)))
  plt.show()

def save_model(sess, saver, mnist, enc_layer_2, x):
  saver.save(sess, FLAGS.save_dir + '/model.ckpt')
  # Dump hidden layer vectors
  with open(FLAGS.data_dir + '/hidden_train', 'wb') as f:
    p = cPickle.Pickler(f)
    for im in mnist.train.images:
      p.dump(sess.run(enc_layer_2, feed_dict={x: [im]}))
  with open(FLAGS.data_dir + '/hidden_test', 'wb') as f:
    p = cPickle.Pickler(f)
    for im in mnist.test.images:
      p.dump(sess.run(enc_layer_2, feed_dict={x: [im]}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory for saving model')
  parser.add_argument('--save_reconstructions', type=str, default=None,
                      help='Location for saving reconstructions')
  parser.add_argument('--load_model', type=str, default=None,
                      help='Directory from which to load model')
  parser.add_argument('--load_samples', type=str, default=None,
                      help='Directory from which to load pickled samples')
  parser.add_argument('--plot', default=False, action='store_true',
                      help='Plot reconstructions if set')
  
  FLAGS = parser.parse_args()
  np.random.seed(1234)
  tf.app.run()

