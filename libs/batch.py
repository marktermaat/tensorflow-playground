class Batch(object):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.idxs = np.random.permutation(range(len(data)))

	def next_batch(batch_size):
		n_batches = len(self.idxs) // batch_size
		for batch_i in range(n_batches):
			idxs_i = self.idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
			yield data[idxs_i], labels[idxs_i]
