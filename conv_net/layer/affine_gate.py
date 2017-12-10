import numpy as np


class AffineGate(object):
    def __init__(self):
        self.input = None
        self.output = None
        self.weight = None
        self.bias = None

    def forward_pass(input, weight, bias):
        self.input = input
        self.weight = weight
        self.bias = bias

        D = np.prod(input.shape[1:])
        input_tf = input.reshape(input.shape[0], D)
        output = np.dot(input_tf, weight) + bias

        return output


    def backward_pass(grad_out):
        """
        Args:
            grad_out: Upstream derivative of shape (N, M)

        Returns:
            grad_in: Gradients of L with respect to input matrix, of shape (N, d_1, ..., d_k)
            grad_weight: Gradient of L with respect to weight matrix of shape (D, M)
            grad_bias: Gradient of L with respect to bias vector of shape (M,)
        """
        if self.input and self.weight and self.bias:
            D = np.prod(self.input.shape[1:])
            input_tf = self.input.reshape(self.input.shape[0], D)

            grad_weight = np.dot(input_tf.T, grad_out)
            grad_in = np.dot(grad_out, self.weight.T).reshape(self.input.shape)
            grad_bias = np.sum(grad_out.T, axis=1)

            return grad_in, grad_weight, grad_bias
