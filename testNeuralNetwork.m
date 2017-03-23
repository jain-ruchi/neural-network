function [ acc ] = testNeuralNetwork(images, labels, W1, W2)
% Test Neural Network (1 Hidden Layer)
% Input:
% images: data matrix
% labels: labels vector/matrix
% W1, W2: weights of the 1 hidden layer neural network
%
% Output:
% acc: accuracy of the neural network for the dataset

[BATCH_SIZE, m] = size(images);

a1 = transpose(images);
a2 = tanh(W1 * [ones(1,BATCH_SIZE); a1]);
a3 = sigmoid(W2 * [ones(1,BATCH_SIZE); a2]);
prediction = bsxfun(@eq, a3, max(a3));
accu_sum = sum(all(prediction == transpose(labels)));
acc = accu_sum / BATCH_SIZE;

end