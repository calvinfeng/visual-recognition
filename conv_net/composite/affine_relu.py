import numpy as np
from conv_net.layer.affine import AffineLayer
from conv_net.layer.relu import ReLULayer


class AffineReLULayer(object):
    def __init__(self):
        self.affine_layer = AffineLayer()
        self.relu_layer = ReLULayer()

    def forward_pass(self, input, weight, bias):
        affine_out = self.affine_layer.forward_pass(input, weight, bias)
        relu_out = self.relu_layer.forward_pass(affine_out)
        return relu_out

    def backward_pass(self, grad_out):
        grad_relu = self.relu_layer.backward_pass(grad_out)
        grad_in, grad_weight, grad_bias = self.affine_layer.backward_pass(grad_relu)
        return grad_in, grad_weight, grad_bias
