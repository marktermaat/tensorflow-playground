import numpy as np
from libs import utils

def load_csv_data(filename):
	data = np.genfromtxt(filename,delimiter=',')
	singular_labels = data[:,0].astype(np.int32)
	data = data[:,1:]

	# Change labels to one-hot encoding
	labels = utils.to_one_hot_encoding(singular_labels, 2)

	# Normalize data
	data = utils.normalize(data)

	# Split train/test data
	train_data, train_labels, test_data, test_labels = utils.split_train_test(data, labels, 10)
	return train_data, train_labels, test_data, test_labels

def load_activity_data(filename):
	data = np.genfromtxt(filename,delimiter=',')
	singular_labels = data[:,0].astype(np.int32)
	data = data[:,1:]

	# Change labels to one-hot encoding
	labels = utils.to_one_hot_encoding(singular_labels, 2)

	# Normalize data
	history = data[:, 0:480-96]
	history_means = np.mean(np.reshape(history, (len(history) * 4, 96)), axis=0)
	history_std = np.std(np.reshape(history, (len(history) * 4, 96)), axis=0)
	data = (data[:, 480-96:] - history_means) / history_std

	# Split train/test data
	train_data, train_labels, test_data, test_labels = utils.split_train_test(data, labels, 10)
	return train_data, train_labels, test_data, test_labels

class Batch(object):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.idxs = np.random.permutation(range(len(data)))

	def next_batch(self, batch_size):
		n_batches = len(self.idxs) // batch_size
		for batch_i in range(n_batches):
			idxs_i = self.idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
			yield self.data[idxs_i], self.labels[idxs_i]
