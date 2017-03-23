function result = sigmoid(z)
% Applies the sigmoid function on all elements of z
% Input:
% z: vector
%
% Output:
% result: vector of same size as input

result = bsxfun(@times, -1, z);
result = bsxfun(@plus, 1, exp(result));
result = bsxfun(@rdivide, 1, result);

result(result<1e-16) = 1e-16;

return
end