import numpy as np


class ReLULayer(object):
    def __init__(self):
        self.input = None

    def forward_pass(self, input):
        """
        Args:
            input: A matrix of any shape

        Returns:
            output: A matrix with ReLU applied, same shape as input
        """
        self.input = input
        return np.maximum(0, self.input)

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out: Upstream derivative

        Returns:
            grad_in: Gradient of upstream variable with respect to input
        """
        grad_in = grad_out
        grad_in[self.input<=0] = 0
        return grad_in
