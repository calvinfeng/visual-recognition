from composite.affine_relu import AffineReLU
from layer.affine import Affine
from gradient_check import eval_numerical_gradient, rel_error
import numpy as np


class FCNetworkModel(object):
    """FCNetworkModel implements a fully connected neural network with arbitrary number of hidden layer.

    Depending on whether batch normalization or dropout is used, every hidden layer is consisted of affine transformation,
    batch normalization, and ReLU activations. The final layer is affine transformation and softmax probability computation.
    """

    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False):
        self.reg = reg
        self.num_layers = len(hidden_dims) + 1
        self.dtype = np.float32
        self.use_batchnorm = use_batchnorm
        self.params = dict()
        self.layers = dict()

        l = 1
        prev_dim = input_dim
        for dim in hidden_dims:
            self.params['W' + str(l)] = np.random.normal(0, scale=weight_scale, size=(prev_dim, dim))
            self.params['b' + str(l)] = np.zeros(dim,)
            self.layers[str(l)] = AffineReLU()

            prev_dim = dim
            l += 1

        self.params['W' + str(l)] = np.random.normal(0, scale=weight_scale, size=(prev_dim, num_classes))
        self.params['b' + str(l)] = np.zeros(num_classes,)
        self.layers[str(l)] = Affine()

    def gradient_check(self, x, y):
        print "Running numeric gradient check with reg = %s" % self.reg
        loss, grads = self.loss(x, y)
        for param_key in sorted(self.params):
            f = lambda _: self.loss(x, y)[0]
            num_grad = eval_numerical_gradient(f, self.params[param_key], verbose=False)
            print "%s relative error: %.2e" % (param_key, rel_error(num_grad, grads[param_key]))

    def loss(self, x, y=None):
        """Computes loss and gradients if label data is provided, else returns scores

        Args:
            x: Input matrix of any shape, depending on initialization values of the model
            y: Labels for training data x, of shape (N,)

        Returns:
            loss: Loss calculation of the model
            scores: Scores of each test example, of shape (N, num_classes)
            grads: Dictionary that contains gradients for each parameter
        """
        x = x.astype(self.dtype)

        mode = 'train'
        if y is None:
            mode = 'test'

        prev_output = x
        l = 1
        while l <= self.num_layers:
            weight = self.params['W' + str(l)]
            bias = self.params['b' + str(l)]
            gate = self.layers[str(l)]

            prev_output = gate.forward_pass(prev_output, weight, bias)
            l += 1

        scores = prev_output
        if mode == 'test':
            return scores

        loss, grad_input = self._softmax(scores, y)
        # Think of scores is an input to the softmax loss, so the gradient returned from _softmax is the gradient of the
        # input, i.e. grad_score

        l = 1
        while l <= self.num_layers:
            weight = self.params['W' + str(l)]
            loss += 0.5 * self.reg * np.sum(weight * weight)
            l += 1

        grads = dict()
        l = self.num_layers
        while l > 0:
            gate = self.layers[str(l)]
            if l == self.num_layers:
                grad_input, grads['W' + str(l)], grads['b' + str(l)] = gate.backward_pass(grad_input)
                grads['W' + str(l)] += self.reg * self.params['W' + str(l)]
            else:
                if self.use_batchnorm:
                    print "Using batchnorm, TO BE IMPLEMENTED"
                else:
                    grad_input, grads['W' + str(l)], grads['b' + str(l)] = gate.backward_pass(grad_input)
                    grads['W' + str(l)] += self.reg * self.params['W' + str(l)]
            l -= 1

        return loss, grads

    def _softmax(self, x, y):
        """Computes the loss and gradient for softmax classification

        Args:
            x: Input data, of shape (N, C) where x[i, j] is the score for jth class for the ith input
            y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

        Returns:
            loss: Scalar value of the loss
            grad_x: Gradient of the loss with respect to x
        """
        # Ensure numerical stability by shifting the input matrix by its largest value in each row.
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)

        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N

        probs = np.exp(log_probs)
        grad_x = probs.copy()
        grad_x[np.arange(N), y] -= 1
        grad_x /= N

        return loss, grad_x
