from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import random
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding='SAME')

def main(_):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None,10])

    #First convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Second convolutional layer
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    #Densely connected layer
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    #Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    #Readout
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.set_random_seed(1234)
        if FLAGS.load_model:
            saver.restore(sess, FLAGS.load_model)
            samples = get_samples()
            sample_predictions = []
            predictions = sess.run(y_conv, feed_dict={x: samples, 
                                                      keep_prob: 1.0})
            sample_predictions = zip(samples, predictions)
            plot(sample_predictions)
            if FLAGS.save_predictions:
                with open(FLAGS.save_predictions, 'wb') as f:
                    cPickle.dump(sample_predictions, f)
        else:
            mnist = input_data.read_data_sets('MNISt_Data', one_hot=True)
            sess.run(tf.initialize_all_variables())
            for i in range(20000):
                batch = mnist.train.next_batch(50)
                if i%100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            save_model(sess, saver)

def plot(sample_predictions):
    reconstruct_count = 20
    fig, ax = plt.subplots(1, reconstruct_count, figsize=(1.2*reconstruct_count, 3))
    for index, im in enumerate(
            random.sample(sample_predictions, reconstruct_count)):
        label = np.argmax(im[1])
        ax[index].set_title('{} ({:.3f})'.format(label, im[1][label]))
        ax[index].imshow(np.reshape(im[0], (28, 28)))
    plt.show()


def save_model(sess, saver):
    saver.save(sess, FLAGS.save_dir + '/model.ckpt')

def get_samples():
    samples = []
    with open(FLAGS.load_samples, 'rb') as f:
        samples = cPickle.load(f)
    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    parser.add_argument('--save_dir', type=str, default='/tmp/data',
                        help='Directory for saving model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Directory from which to load model')
    parser.add_argument('--load_samples', type=str, default=None,
                        help='Location from which to load pickled samples')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Location to pickle predictions')
    FLAGS = parser.parse_args()
    np.random.seed(1234)
    random.seed(1234)
    tf.app.run()
