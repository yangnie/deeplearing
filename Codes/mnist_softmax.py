"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import argparse
import sys
import os
import glob

import logging
import logging.handlers

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '../models/'
log_path = '../logs/'
FLAGS = None
global_step=1500
force = False
file_name = 'softmax_model'
model_filename = os.path.join(path, file_name + '.ckpt-'+str(global_step))

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

LOG_FILENAME = log_path + 'logging_' + file_name + '.out'
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
filehandler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=2000, backupCount=5)
filehandler.setFormatter(logFormatter)
my_logger.addHandler(filehandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
my_logger.addHandler(consoleHandler)

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  saver = tf.train.Saver()
  modelflie_flag = len(glob.glob(model_filename + '.*'))
  my_logger.debug("Model file exists number = %s" % modelflie_flag)
  start=datetime.now()
  if force or not modelflie_flag:
      # Train
      my_logger.debug("Run training.")
      for _ in range(global_step):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  else:
      saver.restore(sess, model_filename)
      my_logger.debug("Model restored.")

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  stop = datetime.now()
  my_logger.debug('Duration: {}'.format(stop-start))
  my_logger.debug(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

  if force or not modelflie_flag:
      save_path = saver.save(sess, model_filename)
      my_logger.debug("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

