%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets

# Load data
data, labels, test_data, test_labels = datasets.load_activity_data('data/insemination_quarter_data.csv')
n_features = data.shape[1]

utils.reset()

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder("float")

# Define network
# Y_pred = utils.create_fully_connected_layer(X, 2, name="layer1", activation=tf.nn.softmax)

# Multilayer:
X_norm = utils.create_batch_normalization_layer(X, is_training, scope='input')
layer1 = utils.create_fully_connected_layer(X_norm, 256, name="layer1")
layer1_norm = utils.create_batch_normalization_layer(layer1, is_training, scope='layer1')
layer2 = utils.create_fully_connected_layer(layer1_norm, 256, name="layer2")
layer2_norm = utils.create_batch_normalization_layer(layer2, is_training, scope='layer2')

# dropout
layer_dropout = tf.nn.dropout(layer2_norm, keep_prob)

Y_pred = utils.create_fully_connected_layer(layer_dropout, 2, name="layer5", activation=tf.nn.softmax)

cross_entropy = utils.cross_entropy_cost_function(Y, Y_pred)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy + l2_loss)
accuracy = utils.create_accuracy_tensor(Y, Y_pred)

predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)

n_epochs = 300
batch_size = 100
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for epoch in range(n_epochs):
		batch = datasets.Batch(data, labels)
		for b_data, b_labels in batch.next_batch(batch_size):
			sess.run(optimizer, feed_dict={X: b_data, Y: b_labels, is_training: True, keep_prob: 0.6})

		training_accuracy = sess.run(accuracy, feed_dict={X: data, Y: labels, is_training: False, keep_prob: 1.0})
		eval_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels, is_training: False, keep_prob: 1.0})
		print(epoch, training_accuracy, eval_accuracy)

	correct_predictions = sess.run(correct_prediction, feed_dict={X: test_data, Y: test_labels, is_training: False, keep_prob: 1.0})

