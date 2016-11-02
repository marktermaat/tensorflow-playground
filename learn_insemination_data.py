%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets

# Load data
data, labels, test_data, test_labels = datasets.load_activity_data('data/insemination_quarter_data_small.csv')
n_features = data.shape[1]

utils.reset()

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')

# Define network
layer1 = utils.create_fully_connected_layer(X, 32, name="layer1")
Y_pred = utils.create_fully_connected_layer(layer1, 2, name="softmax", activation=tf.nn.softmax)

cross_entropy = utils.cross_entropy_cost_function(Y, Y_pred)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
accuracy = utils.create_accuracy_tensor(Y, Y_pred)

predicted_y = tf.argmax(Y_pred, 1)
actual_y = tf.argmax(Y, 1)
correct_prediction = tf.equal(predicted_y, actual_y)

n_epochs = 300
batch_size = 10
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for epoch in range(n_epochs):
		batch = datasets.Batch(data, labels)
		for b_data, b_labels in batch.next_batch(batch_size):
			sess.run(optimizer, feed_dict={X: b_data, Y: b_labels})
			print(b_data)
			print(b_labels)
			break
		break
		training_accuracy = sess.run(accuracy, feed_dict={X: data, Y: labels})
		eval_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_labels})
		print(epoch, training_accuracy, eval_accuracy)

	correct_predictions = sess.run(correct_prediction, feed_dict={X: test_data, Y: test_labels, is_training: False, keep_prob: 1.0})

