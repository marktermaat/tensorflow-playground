%matplotlib osx

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets, streams

utils.reset()

def create_model(input, reuse=None):
	layer1 = utils.create_fully_connected_layer(input, 256, name="layer1", reuse=reuse)
	layer2 = utils.create_fully_connected_layer(layer1, 256, name="layer2", reuse=reuse)
	Y_pred = utils.create_fully_connected_layer(layer2, 2, name="softmax_layer", reuse=reuse, activation=tf.nn.softmax)

	return Y_pred

def create_conv_model(input, n_features, reuse=None):
	filter_size = 5
	n_filters_in_1 = 1
	n_filters_out_1 = 16
	n_filters_in_2 = 16
	n_filters_out_2 = 16

	input4D = tf.reshape(input, [-1, 1, n_features, 1])
	conv1 = utils.create_convolution_layer_1D(input4D, filter_size, n_filters_in_1, n_filters_out_1, name="convolution_1", strides=[1,1,2,1], reuse=reuse)
	# pool1 = utils.create_max_pool_layer(conv1, strides=2)
	conv2 = utils.create_convolution_layer_1D(conv1, filter_size, n_filters_in_2, n_filters_out_2, name="convolution_2", strides=[1,1,2,1], reuse=reuse)
	# pool2 = utils.create_max_pool_layer(conv2, strides=2)

	new_size = int(n_features/2/2) * n_filters_out_2
	conv_output_flat = tf.reshape(conv2, [-1, new_size])

	layer1 = utils.create_fully_connected_layer(conv_output_flat, 64, name="layer1", reuse=reuse)
	layer2 = utils.create_fully_connected_layer(layer1, 64, name="layer2", reuse=reuse)
	Y_pred = utils.create_fully_connected_layer(layer2, 2, name="softmax_layer", reuse=reuse, activation=tf.nn.softmax)

	return Y_pred


n_features = 96
n_epochs = 100
batch_size = 100

# Load data
train_records = streams.create_csv_reader(['data/insemination_quarter_data_train.csv'], 481, n_epochs)
record_data, record_labels = streams.normalize_last_day(train_records)
batch_data, batch_labels = streams.create_batch(record_data, record_labels, batch_size, num_threads=5)

# Create optimizer
Y_pred = create_conv_model(batch_data, n_features)
cross_entropy = utils.cross_entropy_cost_function(batch_labels, Y_pred)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
train_accuracy = utils.create_accuracy_tensor(batch_labels, Y_pred)

# Create dev network
X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
dev_data, dev_labels = datasets.load_dev_data()
Y_pred_dev = create_conv_model(X, n_features, reuse=True)
dev_accuracy = utils.create_accuracy_tensor(Y, Y_pred_dev)

# Tensorboard summaries
# tf.scalar_summary('training accuracy', train_accuracy)
tf.scalar_summary('dev accuracy', dev_accuracy)
merged_summaries = tf.merge_all_summaries()

# Initialize graph
sess = tf.Session()
init_all = tf.initialize_all_variables()
init_local = tf.initialize_local_variables()
sess.run(init_all)
sess.run(init_local)

# Summary writers
# train_writer = tf.train.SummaryWriter('logs/train', sess.graph)
# test_writer = tf.train.SummaryWriter('logs/test')
summary_writer = tf.train.SummaryWriter('logs/c1s5n16_c2s5n16_fc64_fc64')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print("Training")
try:
	counter = 0
	blocksize = 1000
	accuracies = np.zeros(blocksize)
	while not coord.should_stop():
		if(counter % blocksize == 0 and counter != 0):
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
			run_metadata = tf.RunMetadata()

			_, acc, dev_acc, summary = sess.run([optimizer, train_accuracy, dev_accuracy, merged_summaries], feed_dict={X: dev_data, Y: dev_labels}, options=run_options, run_metadata=run_metadata)
			accuracies[counter % blocksize] = acc
			avg_accuracy = np.average(accuracies)
			print(avg_accuracy, dev_acc)
			accuracies = np.zeros(blocksize)

			summary_writer.add_run_metadata(run_metadata, 'step%d' % counter)
			summary_writer.add_summary(summary, counter)
			summary_writer.add_summary(utils.make_summary('Training accuracy', avg_accuracy), counter)
		else:
			_, acc = sess.run([optimizer, train_accuracy], feed_dict={X: dev_data, Y: dev_labels})
			accuracies[counter % blocksize] = acc

		counter += 1
		
except tf.errors.OutOfRangeError:
	print('Done training -- epoch limit reached')
finally:
	# When done, ask the threads to stop.
	# train_writer.close()
	# test_writer.close()
	summary_writer.close()
	coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
