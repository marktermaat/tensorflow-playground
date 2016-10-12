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

X_tensor = tf.reshape(X, [-1, n_features, 1, 1])

filter_size = 3
n_filters_in = 1
n_filters_out = 8
h_1 = utils.create_convolution_layer_1D(X_tensor, filter_size, n_filters_in, n_filters_out, name="convolution_1", strides=[1,2,1,1])
h_2 = utils.create_convolution_layer_1D(h_1, filter_size, 8, 8, name="convolution_2", strides=[1,2,1,1])
h_3 = utils.create_convolution_layer_1D(h_2, filter_size, 8, 8, name="convolution_3", strides=[1,2,1,1])

new_size = int(n_features / 8) * n_filters_out
h_1_flat = tf.reshape(h_3, [-1, new_size])

layer1 = utils.create_fully_connected_layer(h_1_flat, 64, name="fully_connected_layer1")
Y_pred = utils.create_fully_connected_layer(layer1, 2, name="fully_connected_layer2", activation=tf.nn.softmax)

# Define cost and optimizer
cross_entropy = utils.binary_cross_entropy_cost_function(Y, Y_pred)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy + l2_loss)

# Define accuracy (not used during training, only to view statistics)
accuracy = utils.create_accuracy_tensor(Y, Y_pred)

n_epochs = 100
batch_size = 50

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for epoch in range(n_epochs):
	batch = datasets.Batch(data, labels)
	for b_data, b_labels in batch.next_batch(batch_size):
		sess.run(optimizer, feed_dict={X: b_data, Y: b_labels})

	training_accuracy = sess.run(accuracy, feed_dict={X: data, Y: labels})
	eval_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels})
	print(epoch, training_accuracy, eval_accuracy)
