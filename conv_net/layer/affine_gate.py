import numpy as np


class AffineGate(object):
    def __init__(self):
        self.input = None
        self.weight = None
        self.bias = None

    def forward_pass(self, input, weight, bias):
        """
        Args:
            input: A matrix of any shape

        Returns:
            output: A matrix of product of the input and weights plus bias
        """
        self.input = input
        self.weight = weight
        self.bias = bias

        D = np.prod(input.shape[1:])
        input_tf = input.reshape(input.shape[0], D)
        output = np.dot(input_tf, weight) + bias

        return output

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out: Upstream derivative

        Returns:
            grad_in: Gradients of upstream variable with respect to input matrix
            grad_weight: Gradient of upstream variable with respect to weight matrix of shape (D, M)
            grad_bias: Gradient of upstream variable with respect to bias vector of shape (M,)

        The shape changes depending on which layer this gate is inserted. For example, if it is the first gate in the
        network, then grad_in has the shape (N, d_1, ..., d_k) and grad_weight has (D, M). Otherwise, the grad_in would
        be (N x M) and grad_weight would be (M x M).
        """
        if self.input is not None and self.weight is not None:
            D = np.prod(self.input.shape[1:])
            input_tf = self.input.reshape(self.input.shape[0], D)

            grad_weight = np.dot(input_tf.T, grad_out)
            grad_in = np.dot(grad_out, self.weight.T).reshape(self.input.shape)
            grad_bias = np.sum(grad_out.T, axis=1)

            return grad_in, grad_weight, grad_bias
