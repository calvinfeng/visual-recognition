from layer.affine_relu_gate import AffineReLUGate
from layer.affine_gate import AffineGate
import numpy as np


class Network(object):
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10, weight_scale=1e-3, reg=0.0):
        self.reg = reg
        self.num_layers = len(hidden_dims) + 1
        self.dtype = np.float32
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


if __name__ == "__main__":
    hidden_dims = [100, 100, 100, 100]
    rand_input = np.random.randn(100, 32, 32, 3)
    network = Network(hidden_dims)
    print network.loss(rand_input)
