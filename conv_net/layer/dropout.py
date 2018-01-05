import numpy as np


class Dropout(object):
    """Dropout implements a network layer that performs drop out regularization
    """

    def __init__(self):
        self.mask = None
        self.prob = None
        self.mode = None
        self.seed = None

    # TODO: Switch this to keyword arguments
    def forward_pass(self, x, dropout_param):
        """
        Args:
            x: Input of any shape
            dropout_param: A dictionary with the following keys:
                - p: The probability for each neuron dropout
                - mode: 'test' or 'train'
                - seed: Seed for random number generator
        Returns:
            out: Output of the same shape as input
        """
        self.prob, self.mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            self.seed = dropout_param['seed']
            np.random.seed(self.seed)

        if self.mode == 'train':
            self.mask = np.ones(x.shape)
            prob_arr = np.random.random(x.shape)
            self.mask[prob_arr <= self.prob] = 0
            out = self.mask * x
        elif self.mode == 'test':
            out = x
        else:
            raise ValueError("Invalid forward drop out mode: %s" % mode)

        out = out.astype(x.dtype, copy=False)
        return out

    def backward_pass(self, grad_out):
        """
        Args:
            grad_out: Upstream gradient
        Returns:
            grad_x: Gradient w.r.t. input x
        """
        if self.mode == 'train':
            grad_out[self.mask == 0] = 0
            grad_x = grad_out
        elif self.mode == 'test':
            grad_x = grad_out
        else:
            raise ValueError("Invalid backward drop out mode: %s" % self.mode)

        return grad_x
