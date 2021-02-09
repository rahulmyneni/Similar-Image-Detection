# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 10:48:12 2016
Convolutional neural network is applied to classify alphabets (A-J) from notMNINST
Input image resolution is defined as 28x28 and is treated as grayscale (channel =1).
@author: Jae
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

# Import data from 'notMNIST.pickle' which was assembled by "importData.py"

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
  
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import numpy as np

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
  
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
beta_regul = 1e-3
drop_out = 0.5

# TensorFlow setup

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  
  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth*2], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth*2]))
  
  size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2
  layer3_weights = tf.Variable(tf.truncated_normal([size3 * size3 * depth*2, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  
  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  
  # Model.
  def model(data, keep_prob):
    # C1 input 28 x 28x1
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    
    # P1 input 24 x 24 x16
    #pool1 = tf.nn.avg_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    
    # C2 input 12x12x16
    conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='VALID')
    hidden2 = tf.nn.relu(conv2 + layer2_biases)
    
    # P2 input 8 x8x32
    #pool2 = tf.nn.avg_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    
    # F6 input 4x4x32
    shape = pool2.get_shape().as_list()
    reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    
    drop4 = tf.nn.dropout(hidden3, keep_prob)

    return tf.matmul(drop4, layer4_weights) + layer4_biases
  
  # Training computation.
  labels = tf_train_labels
  logits = model(tf_train_dataset,drop_out)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)))
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset,1.0))
  test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))
  
  num_steps = 120001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))