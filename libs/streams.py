import tensorflow as tf

def create_csv_reader(filenames, n_columns, num_epochs, shuffle=True):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[0.1 for col in range(1)] for row in range(n_columns)]
	record_defaults[0][0] = 1

	return tf.decode_csv(value, record_defaults=record_defaults)

def normalize_last_day(records, normalize=None):
	history_data = tf.pack(records[1:-96])
	history_means, history_var = tf.nn.moments(tf.reshape(history_data, [-1, 96]), [0])
	now_data = tf.pack(records[-96:])
	history_std = tf.add(tf.sqrt(history_var), 1e-12)
	data = tf.div(tf.sub(now_data, history_means), history_std)

	if normalize is not None:
		mean, var = tf.nn.moments(data, [0])
		std = tf.add(tf.sqrt(var), 1e-12)
		data = tf.div(tf.sub(data, mean), std)
	
	raw_label = records[0]
	labels = tf.one_hot(records[0], 2)

	return data, labels

# min_after_dequeue defines how big a buffer we will randomly sample from, 
# 	bigger means better shuffling but slower start up and more memory used.
def create_batch(data, labels, batch_size, min_after_dequeue=1000, num_threads=1):
	capacity = min_after_dequeue + 3 * batch_size

	batch_data, batch_labels = tf.train.shuffle_batch(
		[data, labels],
		batch_size=batch_size,
		capacity=capacity,
	  	min_after_dequeue=min_after_dequeue,
	  	allow_smaller_final_batch=True,
	  	num_threads=num_threads)
	return batch_data, batch_labels