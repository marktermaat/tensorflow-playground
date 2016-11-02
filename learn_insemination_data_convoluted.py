# Start python with: ipython
%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from libs import utils, datasets

print("Loading data... ")
data, labels, test_data, test_labels = datasets.load_activity_data('data/insemination_quarter_data_small.csv')
print("Done")

n_features = data.shape[1]

utils.reset()

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
is_training = tf.placeholder(tf.bool, name='is_training')

X_tensor = tf.reshape(X, [-1, n_features, 1, 1])

filter_size = 5
n_filters_in_1 = 1
n_filters_out_1 = 16
n_filters_in_2 = 16
n_filters_out_2 = 32
conv1 = utils.create_convolution_layer_1D(X_tensor, filter_size, n_filters_in_1, n_filters_out_1, name="convolution_1", strides=[1,1,1,1])
pool1 = utils.create_max_pool_layer(conv1, strides=2)
conv2 = utils.create_convolution_layer_1D(pool1, filter_size, n_filters_in_2, n_filters_out_2, name="convolution_2", strides=[1,1,1,1])
pool2 = utils.create_max_pool_layer(conv2, strides=2)

new_size = int(n_features / 2) * n_filters_out
conv_output_flat = tf.reshape(pool2, [-1, new_size])

layer1 = utils.create_fully_connected_layer(conv_output_flat, 64, name="fully_connected_layer1")
Y_pred = utils.create_fully_connected_layer(layer1, 2, name="fully_connected_layer2", activation=tf.nn.softmax)

# Define cost and optimizer
cross_entropy = utils.binary_cross_entropy_cost_function(Y, Y_pred)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# Define accuracy (not used during training, only to view statistics)
accuracy = utils.create_accuracy_tensor(Y, Y_pred)

n_epochs = 100
batch_size = 10

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for epoch in range(n_epochs):
	batch = datasets.Batch(data, labels)
	for b_data, b_labels in batch.next_batch(batch_size):
		sess.run(optimizer, feed_dict={X: b_data, Y: b_labels, is_training: True})

	training_accuracy = sess.run(accuracy, feed_dict={X: data, Y: labels, is_training: False})
	eval_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels, is_training: False})
	print(epoch, training_accuracy, eval_accuracy)
