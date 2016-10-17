# Start python with: ipython
%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from libs import utils, datasets

data, labels, test_data, test_labels = datasets.load_csv_data('data/insemination_quarter_data.csv')
n_features = data.shape[1]

utils.reset()

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
is_training = tf.placeholder(tf.bool, name='is_training')

X_tensor = tf.reshape(X, [-1, n_features, 1, 1])

filter_size = 3
n_filters_in = 1
n_filters_out = 8
X_tensor_norm = utils.create_batch_normalization_layer(X_tensor, is_training, scope='input')
conv1 = utils.create_convolution_layer_1D(X_tensor_norm, filter_size, n_filters_in, n_filters_out, name="convolution_1", strides=[1,2,1,1])
conv1_norm = utils.create_batch_normalization_layer(conv1, is_training, scope='conv_1')
# conv2 = utils.create_convolution_layer_1D(conv1_norm, filter_size, 8, 8, name="convolution_2", strides=[1,2,1,1])
# conv2_norm = utils.create_batch_normalization_layer(conv2, is_training, scope='conv_2')
# conv3 = utils.create_convolution_layer_1D(conv2_norm, filter_size, 8, 8, name="convolution_3", strides=[1,2,1,1])
# conv3_norm = utils.create_batch_normalization_layer(conv3, is_training, scope='conv_3')

new_size = int(n_features / 2) * n_filters_out
conv_output_flat = tf.reshape(conv1_norm, [-1, new_size])

layer1 = utils.create_fully_connected_layer(conv_output_flat, 64, name="fully_connected_layer1")
layer1_norm = utils.create_batch_normalization_layer(layer1, is_training, scope='layer_1')
Y_pred = utils.create_fully_connected_layer(layer1_norm, 2, name="fully_connected_layer2", activation=tf.nn.softmax)

# Define cost and optimizer
cross_entropy = utils.binary_cross_entropy_cost_function(Y, Y_pred)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# Define accuracy (not used during training, only to view statistics)
accuracy = utils.create_accuracy_tensor(Y, Y_pred)

n_epochs = 100
batch_size = 50

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for epoch in range(n_epochs):
	batch = datasets.Batch(data, labels)
	for b_data, b_labels in batch.next_batch(batch_size):
		sess.run(optimizer, feed_dict={X: b_data, Y: b_labels, is_training: True})

	training_accuracy = sess.run(accuracy, feed_dict={X: data, Y: labels, is_training: False})
	eval_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels, is_training: False})
	print(epoch, training_accuracy, eval_accuracy)
