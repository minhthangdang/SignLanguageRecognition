import numpy as np
import pandas as pd
import tensorflow as tf
import math

def load_dataset():
	# read train dataset
	train_dataset = pd.read_csv('sign_mnist_train.csv')
	train_set_y_orig = labels = train_dataset['label'].values # train set labels
	# convert Y to (1, m) vector where m is the number of examples
	# train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	train_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the training set
	train_set_x_orig = train_dataset.values # train set features
	# convert X to (n, m) vector where n is number of features, m is number of examples
	train_set_x_orig = train_set_x_orig.T

	# read test dataset
	test_dataset = pd.read_csv('sign_mnist_test.csv')
	test_set_y_orig = test_dataset['label'].values # test set labels
	# convert Y to (1, m) vector where m is the number of examples
	# test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
	test_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the test set
	test_set_x_orig = test_dataset.values # test set features
	# convert X to (n, m) vector where n is number of features, m is number of examples
	test_set_x_orig = test_set_x_orig.T

	classes = np.array(labels)
	classes = np.unique(classes)

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def create_placeholders(n_x, n_y):
	"""
	Creates the placeholders for the tensorflow session.

	Arguments:
	n_x -- scalar, size of an image vector (num_px * num_px = 28 * 28 = 784)
	n_y -- scalar, number of classes (24 classes)

	Returns:
	X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
	Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"

	Tips:
	- Use None because it lets us be flexible on the number of examples we will use for the placeholders.
	  In fact, the number of examples during test/train is different.
	"""

	X = tf.placeholder(shape=(n_x, None), dtype=tf.float32)
	Y = tf.placeholder(shape=(n_y, None), dtype=tf.float32)

	return X, Y

def initialize_parameters():
	"""
	Initializes parameters to build a neural network with tensorflow. The shapes are:
	                    W1 : [25, 784]
	                    b1 : [25, 1]
	                    W2 : [12, 25]
	                    b2 : [12, 1]
	                    W3 : [24, 12]
	                    b3 : [24, 1]

	Returns:
	parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
	"""

	tf.set_random_seed(1)
	    
	W1 = tf.get_variable("W1", [25, 784], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [24, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
	b3 = tf.get_variable("b3", [24, 1], initializer = tf.zeros_initializer())

	parameters = {"W1": W1,
	              "b1": b1,
	              "W2": W2,
	              "b2": b2,
	              "W3": W3,
	              "b3": b3}

	return parameters

def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
	              the shapes are given in initialize_parameters

	Returns:
	Z3 -- the output of the last LINEAR unit
	"""

	# Retrieve the parameters from the dictionary "parameters" 
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']

	######################################################## Numpy Equivalents:
	Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
	A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, A1) + b2
	A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3, A2) + b3

	return Z3

def compute_cost(Z3, Y):
	"""
	Computes the cost

	Arguments:
	Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
	Y -- "true" labels vector placeholder, same shape as Z3

	Returns:
	cost - Tensor of the cost function
	"""

	# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

	return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	"""
	Creates a list of random minibatches from (X, Y)

	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- one-hot matrix
	mini_batch_size - size of the mini-batches, integer

	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""

	m = X.shape[1]                  # number of training examples
	mini_batches = []
	np.random.seed(seed)

	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches