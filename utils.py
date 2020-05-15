import numpy as np
import pandas as pd

def load_dataset():
	# read train dataset
	train_dataset = pd.read_csv('sign_mnist_train.csv')
	train_set_y_orig = train_dataset['label'].values # train set labels
	# convert Y to (1, m) vector where m is the number of examples
	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	train_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the training set
	train_set_x_orig = train_dataset.values # train set features
	# convert X to (m, n) vector where m is number of examples, n is number of features
	train_set_x_orig = train_set_x_orig.T

	# read test dataset
	test_dataset = pd.read_csv('sign_mnist_test.csv')
	test_set_y_orig = test_dataset['label'].values # test set labels
	# convert Y to (1, m) vector where m is the number of examples
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
	test_dataset.drop('label', axis = 1, inplace = True) # drop the label coloumn from the test set
	test_set_x_orig = test_dataset.values # test set features
	# convert X to (n, m) vector where n is number of features, m is number of examples
	test_set_x_orig = test_set_x_orig.T

    #classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

