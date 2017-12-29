from layer.affine_relu_gate import AffineReLUGate
from layer.affine_gate import AffineGate
import numpy as np


class Network(object):
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False):
        self.reg = reg
        self.num_layers = len(hidden_dims) + 1
        self.dtype = np.float32
        self.use_batchnorm = use_batchnorm
        self.params = dict()
        self.gates = dict()

        layer = 1
        prev_dim = input_dim
        for dim in hidden_dims:
            self.params['W' + str(layer)] = np.random.normal(0, scale=weight_scale, size=(prev_dim, dim))
            self.params['b' + str(layer)] = np.zeros(dim,)
            self.gates[str(layer)] = AffineReLUGate()

            prev_dim = dim
            layer += 1

        self.params['W' + str(layer)] = np.random.normal(0, scale=weight_scale, size=(prev_dim, num_classes))
        self.params['b' + str(layer)] = np.zeros(num_classes,)
        self.gates[str(layer)] = AffineGate()

    def loss(self, x, y=None):
        x = x.astype(self.dtype)

        mode = 'train'
        if y is None:
            mode = 'test'

        prev_output = x
        layer = 1
        while layer <= self.num_layers:
            weight = self.params['W' + str(layer)]
            bias = self.params['b' + str(layer)]
            gate = self.gates[str(layer)]

            prev_output = gate.forward_pass(prev_output, weight, bias)
            layer += 1

        scores = prev_output
        if mode == 'test':
            return scores

        loss, grad_input = self._softmax(scores, y)
        # Think of scores is an input to the softmax loss, so the gradient returned from _softmax is the gradient of the
        # input, i.e. grad_score

        layer = 1
        while layer <= self.num_layers:
            weight = self.params['W' + str(layer)]
            loss += 0.5 * self.reg * np.sum(weight * weight)
            layer += 1

        grads = dict()
        layer = self.num_layers
        while layer > 0:
            gate = self.gates[str(layer)]
            if layer == self.num_layers:
                grad_input, grads['W' + str(layer)], grads['b' + str(layer)] = gate.backward_pass(grad_input)
                grads['W' + str(layer)] += self.reg * self.params['W' + str(layer)]
            else:
                if self.use_batchnorm:
                    print "Using batchnorm, TO BE IMPLEMENTED"
                else:
                    grad_input, grads['W' + str(layer)], grads['b' + str(layer)] = gate.backward_pass(grad_input)
                    grads['W' + str(layer)] += self.reg * self.params['W' + str(layer)]
            layer -= 1

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


if __name__ == "__main__":
    N = 100
    num_classes = 10
    hidden_dims = [100, 100, 100, 100]

    rand_inputs = np.random.randn(N, 32, 32, 3)
    rand_labels = np.random.randint(num_classes, size=(N,))

    network = Network(hidden_dims)
    print rand_labels
    scores =  network.loss(rand_inputs)
    loss, grads = network.loss(rand_inputs, rand_labels)

    print "Scores: %s" % str(scores.shape)
    print "Loss: %s and Grads has shape %s" % (loss, str(grads.keys()))
