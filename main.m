% load training set
load('Dataset/train.mat');

% transform training data into matrix
images = train.images;
[s, t, N] = size(images);

NUM_FEATURES = s * t;
train_data = zeros(N, NUM_FEATURES);

for i=1:N
    train_data(i,:) = reshape(images(:,:,i), NUM_FEATURES, 1);
end

% normalize dataset
M = mean(train_data);
S = std(train_data);
S(S == 0) = 1;

train_data = bsxfun(@minus, train_data, M);
train_data = bsxfun(@rdivide, train_data, S);

% one-hot encode training labels
train_label = bsxfun(@eq, train.labels(:), 0:max(train.labels));

% load test set
load('Dataset/test.mat');

images = test.images;
[x, y, z] = size(images);
test_data = zeros(z, NUM_FEATURES);

for i=1:z
   test_data(i,:) = reshape(images(:,:,i), NUM_FEATURES, 1);
end

test_data = bsxfun(@minus, test_data, M);
test_data = bsxfun(@rdivide, test_data, S);
test_label = bsxfun(@eq, test.labels(:), 0:max(test.labels));

[W1, W2] = trainNeuralNetwork(train_data, train_label);
train_accuracy = testNeuralNetwork(train_data, train_label, W1, W2);
test_accuracy = testNeuralNetwork(test_data, test_label, W1, W2);

S = sprintf('Test Accuracy: %d', test_accuracy);
disp(S);