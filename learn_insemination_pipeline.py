%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets, streams

utils.reset()

def create_model(input, reuse=None):
	layer1 = utils.create_fully_connected_layer(input, 256, name="layer1", reuse=reuse)
	layer2 = utils.create_fully_connected_layer(layer1, 256, name="layer2", reuse=reuse)
	layer3 = utils.create_fully_connected_layer(layer2, 256, name="layer3", reuse=reuse)
	Y_pred = utils.create_fully_connected_layer(layer3, 2, name="softmax_layer", reuse=reuse, activation=tf.nn.softmax)

	return Y_pred


n_features = 96
n_epochs = 100
batch_size = 100

# Load data
train_records = streams.create_csv_reader(['data/insemination_quarter_data_train.csv'], 481, n_epochs)
record_data, record_labels = streams.normalize_last_day(train_records)
batch_data, batch_labels = streams.create_batch(record_data, record_labels, batch_size, num_threads=10)

# Create optimizer
Y_pred = create_model(batch_data)
cross_entropy = utils.cross_entropy_cost_function(batch_labels, Y_pred)
optimizer = tf.train.AdagradOptimizer(0.001).minimize(cross_entropy)
train_accuracy = utils.create_accuracy_tensor(batch_labels, Y_pred)

# Create dev network
X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
dev_data, dev_labels = datasets.load_dev_data()
Y_pred_dev = create_model(X, reuse=True)
dev_accuracy = utils.create_accuracy_tensor(Y, Y_pred_dev)

# Initialize graph
sess = tf.Session()
init_all = tf.initialize_all_variables()
init_local = tf.initialize_local_variables()
sess.run(init_all)
sess.run(init_local)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print("Training")
try:
	while not coord.should_stop():
		accuracies = np.zeros(10)
		for i in range(100):
			_, acc = sess.run([optimizer, train_accuracy])
			accuracies[i] = acc

		dev_acc = sess.run(dev_accuracy, feed_dict={X: dev_data, Y: dev_labels})
		print(np.average(accuracies), dev_acc)
except tf.errors.OutOfRangeError:
	print('Done training -- epoch limit reached')
finally:
	# When done, ask the threads to stop.
	coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
