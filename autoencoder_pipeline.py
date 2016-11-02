%matplotlib osx # For plotting

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from libs import utils, datasets, streams

utils.reset()

def create_model(input, reuse=None):
	input_size = input.get_shape().as_list()[1]
	encoding1, weight1 = utils.create_fully_connected_layer_without_bias(input, 64, name="encoding1", reuse=reuse, return_all=True)
	encoding2, weight2 = utils.create_fully_connected_layer_without_bias(encoding1, 32, name="encoding2", reuse=reuse, return_all=True)
	encoding3, weight3 = utils.create_fully_connected_layer_without_bias(encoding2, 16, name="encoding3", reuse=reuse, return_all=True)

	decoding3 = utils.create_fully_connected_layer_without_bias(encoding3, 32, name="decoding3", reuse=reuse, given_weight=tf.transpose(weight3))
	decoding2 = utils.create_fully_connected_layer_without_bias(decoding3, 64, name="decoding2", reuse=reuse, given_weight=tf.transpose(weight2))
	Y = utils.create_fully_connected_layer_without_bias(decoding2, input_size, name="decoding1", reuse=reuse, given_weight=tf.transpose(weight1))

	return Y

def create_conv_model(input, n_features, reuse=None):
	input_size = tf.shape(input)[0]
	input4D = tf.reshape(input, [-1, 1, n_features, 1])
	encoding1, weight1 = utils.create_convolution_layer_1D_without_bias(input4D, 5, 1, 16, name="encoding1", reuse=reuse, return_all=True, strides=[1,1,2,1])
	encoding2, weight2 = utils.create_convolution_layer_1D_without_bias(encoding1, 5, 16, 16, name="encoding2", reuse=reuse, return_all=True, strides=[1,1,2,1])
	encoding3, weight3 = utils.create_convolution_layer_1D_without_bias(encoding2, 5, 16, 16, name="encoding3", reuse=reuse, return_all=True, strides=[1,1,2,1])

	decoding3 = utils.create_convolution_layer_1D_without_bias_transpose(encoding3, weight3, (input_size, 1, int(n_features/4), 16), name="decoding3", strides=[1,1,2,1])
	decoding2 = utils.create_convolution_layer_1D_without_bias_transpose(decoding3, weight2, (input_size, 1, int(n_features/2), 16), name="decoding2", strides=[1,1,2,1])
	decoding1 = utils.create_convolution_layer_1D_without_bias_transpose(decoding2, weight1, (input_size, 1, n_features, 1), name="decoding1", strides=[1,1,2,1])
	Y = tf.reshape(decoding1, [-1, n_features])

	return Y

n_features = 96
n_epochs = 500
batch_size = 100

# Load data
train_records = streams.create_csv_reader(['data/insemination_quarter_data_train.csv'], 481, n_epochs)
record_data, record_labels = streams.normalize_last_day(train_records, normalize=True)
batch_data, batch_labels = streams.create_batch(record_data, record_labels, batch_size, num_threads=5)

# Create optimizer
Y = create_conv_model(batch_data, n_features)
cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(batch_data, Y), 1))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Create dev network
X = tf.placeholder(tf.float32, shape=[None, n_features], name='X')
dev_data, dev_labels = datasets.load_dev_data()
Y_dev = create_conv_model(X, n_features, reuse=True)

# Tensorboard summaries
tf.scalar_summary('Cost', cost)
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
summary_writer = tf.train.SummaryWriter('logs/conv_autoencoder_16x5_16x5_16x5_strides2_4dtensorfix')

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print("Training")
try:
	counter = 0
	blocksize = 1000
	accuracies = np.zeros(blocksize)
	while not coord.should_stop():
		if(counter % blocksize == 0 and counter != 0):
			sess.run(optimizer)
		else:
			_, c, summary, b, y_pred = sess.run([optimizer, cost, merged_summaries, batch_data, Y])
			if c < 1:
				summary_writer.add_summary(summary, counter)
			print(c)
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
