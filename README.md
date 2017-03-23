# A Simple Neural Network
Train and test a Neural Network on the MNIST Databse of handwritten digits with average accuracy of about 95%.

## Requirements:
* GNU Octave or MATLAB 2016a

## The Network:
* One input layer, one hidden layer, and one output layer.
* Hidden layer size = 200 units.
* Output layer size = 10 units (10 digits, i.e., 0, 1, 2, ..., 9).
* Applies nonlinear function ```tanh``` to input layer and ```sigmoid``` to hidden layer.
* Uses cross entropy for error function.

## The Dataset:
* From the [MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/).
* Training set: 60,000 examples
* Test set: 10,000 examples
* Contained in ```.mat``` files in ```/Dataset```.

## Source Files:
* ```trainNeuralNetwork.m```: trains NN and returns two matrices, W1 and W2 containing weights.
* ```testNeuralNetowrk.m```: tests NN on test and returns accuracy.
* ```sigmoid.m```: applies the Sigmoid function on all elments of an array.

## Instructions:
* cd to this directory.
* Run ```main``` from Octave or MATLAB.
* Optionally, uncomment ```line 20```, ```line 47```, and ```line 50``` for visualization.