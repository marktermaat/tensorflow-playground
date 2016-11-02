import numpy as np
import tensorflow as tf

def reset():
	tf.reset_default_graph()
	if 'sess' in locals():
		sess.close()

def to_one_hot_encoding(labels, nr_labels):
	one_hot = np.zeros((labels.size, nr_labels))
	one_hot[np.arange(labels.size), labels] = 1
	return one_hot

def normalize(data):
	return (data - np.mean(data)) / np.std(data)

def get_data_of_label(data, labels, label_column):
	return data[np.where(labels[:,label_column] == 1.0)[0]]

"""Splits the data into a train and a test split_train_test
	split_fraction: the fraction of what should be test data. For example, use '10' to indicate 1/10 of the data for test
"""
def split_train_test(data, labels, test_fraction):
	test_idx = np.random.permutation(range(len(data)))[0:int(len(data)/test_fraction)]
	test_data = data[test_idx]
	test_labels = labels[test_idx]
	train_data = np.delete(data, test_idx, axis=0)
	train_labels = np.delete(labels, test_idx, axis=0)
	return train_data, train_labels, test_data, test_labels

""" This function simply takes the absolute difference between the predicted and the real value"""
def abs_difference_cost_function(Y_tensor, Y_pred_tensor):
	return tf.reduce_mean(tf.abs(Y_pred_tensor - Y_tensor))

""" This function uses the cross entropy, which works best for one-hot encoded labels (with more than 2 labels)"""
def cross_entropy_cost_function(Y_tensor, Y_pred_tensor):
	return -tf.reduce_sum(Y_tensor * tf.log(Y_pred_tensor + 1e-12))

""" The binary cross entropy works best for one-hot encoded labels where there are only 2 possible lables """
def binary_cross_entropy_cost_function(Y_tensor, Y_pred_tensor):
	eps = 1e-12
	return (-(Y_tensor * tf.log(Y_pred_tensor + eps) + (1. - Y_tensor) * tf.log(1. - Y_pred_tensor + eps)))

def create_accuracy_tensor(Y_tensor, Y_pred_tensor):
	predicted_y = tf.argmax(Y_pred_tensor, 1)
	actual_y = tf.argmax(Y_tensor, 1)
	correct_prediction = tf.equal(predicted_y, actual_y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	return accuracy

def create_fully_connected_layer(input, output_size, name=None, activation=tf.nn.relu, is_training=None, reuse=None,
	return_all=None, given_weight=None, given_bias=None):
	input_size = input.get_shape().as_list()[1]
	with tf.variable_scope(name or "fully_connected") as scope:
		if reuse:
			scope.reuse_variables()
		Weight = given_weight if given_weight is not None else tf.get_variable(name='Weight', shape=[input_size, output_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		bias = given_bias if given_bias is not None else tf.get_variable(name='bias', shape=[output_size], initializer=tf.constant_initializer())
		h = tf.matmul(input, Weight) + bias
		if return_all:
			return activation(h), Weight, bias
		else:
			return activation(h)

def create_fully_connected_layer_without_bias(input, output_size, name=None, activation=tf.nn.relu, is_training=None, reuse=None,
	return_all=None, given_weight=None):
	input_size = input.get_shape().as_list()[1]
	with tf.variable_scope(name or "fully_connected") as scope:
		if reuse:
			scope.reuse_variables()
		Weight = given_weight if given_weight is not None else tf.get_variable(name='Weight', shape=[input_size, output_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		h = tf.matmul(input, Weight)
		if return_all:
			return activation(h), Weight
		else:
			return activation(h)

def create_convolution_layer_1D(input, filter_size, n_filters_in, n_filters_out, name=None,
	activation=tf.nn.relu, strides=[1,2,2,1], reuse=None, return_all=None,
	given_weight=None, given_bias=None):
	with tf.variable_scope(name or "convolution") as scope:
		if reuse:
			scope.reuse_variables()
		Weight = given_weight if given_weight is not None else tf.get_variable(name='Weight', shape=[1, filter_size, n_filters_in, n_filters_out], initializer=tf.random_normal_initializer())
		bias = given_bias if given_bias is not None else tf.get_variable( name='bias', shape=[n_filters_out], initializer=tf.constant_initializer())
		h = tf.nn.bias_add( tf.nn.conv2d(input=input, filter=Weight, strides=strides, padding='SAME'), bias)
		if return_all:
			return activation(h), Weight, bias
		else:
			return activation(h)

def create_convolution_layer_1D_without_bias(input, filter_size, n_filters_in, n_filters_out, name=None,
	activation=tf.nn.relu, strides=[1,1,2,1], reuse=None, return_all=None, given_weight=None):
	with tf.variable_scope(name or "convolution") as scope:
		if reuse:
			scope.reuse_variables()
		Weight = given_weight if given_weight is not None else tf.get_variable(name='Weight', shape=[1, filter_size, n_filters_in, n_filters_out], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
		h = tf.nn.conv2d(input, Weight, strides, padding='SAME')
		if return_all:
			return activation(h), Weight
		else:
			return activation(h)

def create_convolution_layer_1D_without_bias_transpose(input, Weight, output_shape, name=None, activation=tf.nn.relu, strides=[1,1,2,1], reuse=None):
	with tf.variable_scope(name or "convolution") as scope:
		if reuse:
			scope.reuse_variables()
		h = tf.nn.conv2d_transpose(input, Weight, output_shape, strides=strides, padding='SAME')
		return activation(h)

def create_max_pool_layer(input, strides=2):
	return tf.nn.max_pool(input, ksize=[1, strides, strides, 1], strides=[1, strides, strides, 1], padding='SAME')

def create_batch_normalization_layer(input, is_training, scope='BN'):
    bn_train = tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None,
    trainable=True,
    scope=scope)
    bn_inference = tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True,
    trainable=True,
    scope=scope)
    layer = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
    return layer

def make_summary(name, val):
	return tf.core.framework.summary_pb2.Summary(value=[tf.core.framework.summary_pb2.Summary.Value(
		tag=name,  simple_value=val)])
