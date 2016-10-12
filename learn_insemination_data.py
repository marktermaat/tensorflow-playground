# Start python with: ipython
%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets

# Load data
# data, labels, test_data, test_labels = datasets.load_csv_data('data/insemination_quarter_data_small.csv')
data, labels, test_data, test_labels = datasets.load_activity_data('data/insemination_quarter_data.csv')
n_features = data.shape[1]

utils.reset()

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
is_training = tf.placeholder(tf.bool, name='is_training')

# Define network
# Y_pred = utils.create_fully_connected_layer(X, 2, name="layer1", activation=tf.nn.softmax)

# Multilayer:
layer1 = utils.create_fully_connected_layer(X, 32, name="layer1")
layer2 = utils.create_fully_connected_layer(layer1, 32, name="layer2")
layer3 = utils.create_fully_connected_layer(layer2, 32, name="layer3")
layer4 = utils.create_fully_connected_layer(layer3, 32, name="layer4")
Y_pred = utils.create_fully_connected_layer(layer4, 2, name="layer5", activation=tf.nn.softmax)

cross_entropy = utils.cross_entropy_cost_function(Y, Y_pred)
# l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), tf.trainable_variables())
# optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy + l2_loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
accuracy = utils.create_accuracy_tensor(Y, Y_pred)

predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)

n_epochs = 300
batch_size = 50
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for epoch in range(n_epochs):
		batch = datasets.Batch(data, labels)
		for b_data, b_labels in batch.next_batch(batch_size):
			sess.run(optimizer, feed_dict={X: b_data, Y: b_labels, is_training: True})

		training_accuracy = sess.run(accuracy, feed_dict={X: data, Y: labels, is_training: False})
		eval_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels, is_training: False})
		print(epoch, training_accuracy, eval_accuracy)

	correct_predictions = sess.run(correct_prediction, feed_dict={X: test_data, Y: test_labels, is_training: False})

	# Print network
	g = tf.get_default_graph()
	W = g.get_tensor_by_name('layer1/Weight:0')
	W_arr = np.array(W.eval(session=sess))
	print(W_arr.shape)

	fig, ax = plt.subplots(1, 2, figsize=(20, 3))
	x = np.linspace(0, n_features, n_features)
	for col_i in range(2):
		print(W_arr[:, col_i].shape)
		ax[col_i].plot(x, W_arr[:, col_i])
