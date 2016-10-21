%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets

utils.reset()

def create_pipeline(filenames, n_columns, batch_size, num_epochs):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[0.1 for col in range(1)] for row in range(n_columns)]
	record_defaults[0][0] = 1

	values = tf.decode_csv(value, record_defaults=record_defaults)
	history_data = tf.pack(values[1:n_columns - 96])
	history_means, history_var = tf.nn.moments(tf.reshape(history_data, [-1, 96]), [0])
	now_data = tf.pack(values[n_columns - 96:])
	history_std = tf.add(tf.sqrt(history_var), 1e-12)
	example_data = tf.div(tf.sub(now_data, history_means), history_std)
	raw_label = values[0]

	example_label = tf.one_hot(values[0], 2)

	# min_after_dequeue defines how big a buffer we will randomly sample
	#   from -- bigger means better shuffling but slower start up and more
	#   memory used.
	min_after_dequeue = 1000
	# capacity must be larger than min_after_dequeue and the amount larger
	#   determines the maximum we will prefetch.  Recommendation:
	#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
	capacity = min_after_dequeue + 3 * batch_size

	batch_data, batch_labels = tf.train.shuffle_batch(
		[example_data, example_label],
		batch_size=batch_size,
		capacity=capacity,
	  	min_after_dequeue=min_after_dequeue,
	  	allow_smaller_final_batch=True)
	return batch_data, batch_labels


def create_model(input, reuse=None):
	layer1 = utils.create_fully_connected_layer(input, 128, name="layer1", reuse=reuse)
	layer2 = utils.create_fully_connected_layer(layer1, 128, name="layer2", reuse=reuse)
	Y_pred = utils.create_fully_connected_layer(layer2, 2, name="softmax_layer", reuse=reuse, activation=tf.nn.softmax)

	return Y_pred


def load_dev_data():
	data = np.genfromtxt('data/insemination_quarter_data_dev.csv', delimiter=',')
	singular_labels = data[:,0].astype(np.int32)
	data = data[:,1:]
	labels = utils.to_one_hot_encoding(singular_labels, 2)

	# Normalize data
	history = data[:, 0:480-96]
	history_means = np.mean(np.reshape(history, (len(history) * 4, 96)), axis=0)
	history_std = np.std(np.reshape(history, (len(history) * 4, 96)), axis=0)
	data = (data[:, 480-96:] - history_means) / history_std

	return data, labels


n_features = 96

train_batch, train_labels = create_pipeline(['data/insemination_quarter_data_train.csv'], n_features * 5 + 1, 100, 100)
dev_data, dev_labels = load_dev_data()

Y_pred = create_model(train_batch)
cross_entropy = utils.cross_entropy_cost_function(train_labels, Y_pred)
optimizer = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)
train_accuracy = utils.create_accuracy_tensor(train_labels, Y_pred)

X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')

Y_pred_dev = create_model(X, reuse=True)
dev_accuracy = utils.create_accuracy_tensor(Y, Y_pred_dev)

sess = tf.Session()
init_all = tf.initialize_all_variables()
init_local = tf.initialize_local_variables()
sess.run(init_all)
sess.run(init_local)

tf.train.start_queue_runners(sess=sess)

while True:
  _, acc = sess.run([optimizer, train_accuracy])
  predicted, wanted, ce = sess.run([Y_pred, train_labels, cross_entropy])
  dev_acc = sess.run(dev_accuracy, feed_dict={X: dev_data, Y: dev_labels})
  print(acc, dev_acc)
