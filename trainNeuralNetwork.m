function [ W1 W2 ] = trainNeuralNetwork( images, labels )
% Train Neural Network (1 Hidden Layer)
% Input:
% images: data matrix
% labels: labels vector/matrix
%
% Output:
% W1, W2: weights of the 1 hidden layer neural network

[N, M] = size(images);

ITER = 2000;
BATCH_SIZE = 32;
INV_B = 1/BATCH_SIZE;
ETA = 0.5 * INV_B;

W1 = randn(200, M + 1)/100; % initialize weights to random small values
W2 = randn(10, 201)/100;

% accuracies = zeros(ITER, 1);

for t=1:ITER
    g1 = zeros(200, M + 1);
    g2 = zeros(10, 201);

    shuffle = randperm(N);
    shuffled_data = images(shuffle,:);
    shuffled_label = labels(shuffle,:);
    batch_data = shuffled_data(1:BATCH_SIZE,:);
    batch_label = shuffled_label(1:BATCH_SIZE,:);

    a1 = transpose(batch_data);
    a2 = tanh(W1 * [ones(1,BATCH_SIZE); a1]);
    a3 = sigmoid(W2 * [ones(1,BATCH_SIZE); a2]);

    % backpropagation
    d3 = a3 - transpose(batch_label); % 10 x BATCH_SIZE
    d2 = (transpose(W2(:,2:201)) * d3) .* (1 - (a2).^2);

    % update g2 matrix
    g2 = d3 * [ones(BATCH_SIZE, 1) transpose(a2)];
    g1 = d2 * [ones(BATCH_SIZE, 1) transpose(a1)];

    W1 = W1 - ETA * g1;
    W2 = W2 - ETA * g2;

    % accuracies(t,:) = testNeuralNetwork(images, labels, W1, W2);
end

% plot(1:ITER, accuracies(:,1));

end