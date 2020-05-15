# Sign Language Recognition
Deep learning for sign language recognition using MNIST image dataset

## Problem Statement

This is my own exercise of Coursera's Improving Deep Neural
Networks: Hyperparameter tuning, Regularization and Optimization which I have
passed (https://www.coursera.org/account/accomplishments/records/KDWADX7V43SS).

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/KDWADX7V43SS.jpg?raw=true" alt="Sign Language" width="400"/><br>

I strongly believe the best way to learn something is to put it in practice, hence
I did this exercise to solidify my knowledge.

In the course, the programming assignment was to implement a deep neural network
model to recognize numbers from 0 to 5 in sign language. Whereas, in this
exercise I adapted the model to recognize 24 classes of letters (excluding J and Z)
in American Sign Language.

The dataset is obtained from Kaggle (https://www.kaggle.com/datamunge/sign-language-mnist).
The training data has 27,455 examples and the test data has 7172 examples. Each
example is a 28x28=784 pixel vector with grayscale values between 0-255.

An illustration of the sign language is shown here (courtesy of Kaggle):

<img src="https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/datasets_3258_5337_amer_sign2.png" alt="Sign Language" width="400"/><br>

Grayscale images with (0-255) pixel values:

<img src="https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/datasets_3258_5337_amer_sign3.png" alt="Sign Language" width="400"/><br>

One example in the MNIST dataset:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/mnist-example.JPG?raw=true" alt="Sign Language" width="400"/><br>


## Deep Learning Model

In this exercise, I keep the same network architecture as the one used in the course.
The model is as follows LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX.

So there are two hidden layers and one output layer. The architecture is depicted below:

<img src="https://raw.githubusercontent.com/minhthangdang/minhthangdang.github.io/master/nn-architecture.JPG" alt="Sign Language" width="400"/><br>

The neural network is implemented in Python and Tensorflow 1.x

I keep all the default hyperparameter values (learning_rate = 0.0001,
  num_epochs = 1500, minibatch_size = 32, etc.)

This is the result:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/cost-function.JPG?raw=true" alt="Sign Language" width="400"/><br>

So you can see that the model is clearly overfitting. Adding regularization methods such as
L2 regularization or dropout can help reduce overfitting, but that's out of the scope
of this exercise. However please stay tuned as more to come.

Should you have any questions, please contact me via Linkedin: https://www.linkedin.com/in/minh-thang-dang/
