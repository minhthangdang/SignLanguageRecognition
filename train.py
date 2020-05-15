import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from utils import load_dataset, predict
from model import model
import scipy
from PIL import Image
from scipy import ndimage

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
#index = 1 # change the index to see another picture
#plt.imshow(X_train_orig[:, index].reshape((28, 28)))
#plt.show()
#print ("y = " + str(np.squeeze(Y_train_orig[index])))

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
# Creates a matrix where the i-th row corresponds to the ith class number and the jth column
# corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
# will be 1.
label_binrizer = LabelBinarizer()
Y_train = label_binrizer.fit_transform(Y_train_orig).T
Y_test = label_binrizer.fit_transform(Y_test_orig).T

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# train the neural network
parameters = model(X_train, Y_train, X_test, Y_test)

fname = "testimage.png"
image = np.array(ndimage.imread(fname, flatten=False, mode='L'))
image = image/255.
my_image = scipy.misc.imresize(image, size=(28, 28)).reshape((1, 28*28)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Predict: y = " + str(np.squeeze(my_image_prediction)))