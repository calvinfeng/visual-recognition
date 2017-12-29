import numpy as np
from affine_gate import AffineGate
from relu_gate import ReLUGate


class AffineReLUGate(object):
    def __init__(self):
        self.affine_gate = AffineGate()
        self.relu_gate = ReLUGate()

    def forward_pass(self, input, weight, bias):
        affine_out = self.affine_gate.forward_pass(input, weight, bias)
        relu_out = self.relu_gate.forward_pass(affine_out)
        return relu_out

    def backward_pass(self, grad_out):
        grad_relu = self.relu_gate.backward_pass(grad_out)
        grad_in, grad_weight, grad_bias = self.affine_gate.backward_pass(grad_relu)
        return grad_in, grad_weight, grad_bias


if __name__ == "__main__":
    N = 100 # Number of examples
    input_dim = 32 * 32 * 3 # Each image is 32 by 32 pixels and each pixel has 3 channels
    hidden_dim = 100

    rand_input = np.random.randn(N, 32, 32, 3) # 5 random images
    rand_weight = np.random.randn(input_dim, hidden_dim)
    rand_bias = np.zeros(hidden_dim,)

    gate = AffineReLUGate()
    print gate.forward_pass(rand_input, rand_weight, rand_bias).shape

    rand_grad_out = np.random.randn(N, hidden_dim)
    grad_in, grad_weight, grad_bias = gate.backward_pass(rand_grad_out)
    print grad_in.shape
    print grad_weight.shape
    print grad_bias.shape
