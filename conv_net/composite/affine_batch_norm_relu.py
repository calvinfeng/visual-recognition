import numpy as np
from conv_net.layer.affine import AffineLayer
from conv_net.layer.relu import ReLULayer
from conv_net.layer.batch_norm import BatchNormLayer


class AffineBatchNormReLULayer(object):
    def __init__(self):
        self.affine_layer = AffineLayer()
        self.relu_layer = ReLULayer()
        self.batch_norm_layer = BatchNormLayer()

    def forward_pass(self, x, w, b, gamma, beta, bn_param):
        """ Performs forward propagation through affine, batch norm, and ReLU layers

        Args:
            x: Input
            w: Weights
            b: Bias
            gamma: Scale factor
            beta: Shifting factor
            bn_param: Batch normalization parameters

        Returns:
            relu_out: Output from ReLU layer
        """
        affine_out = self.affine_layer.forward_pass(x, w, b)
        batch_norm_out = self.batch_norm_layer.forward_pass(affine_out, gamma, beta, bn_param)
        relu_out = self.relu_layer.forward_pass(batch_norm_out)

        return relu_out

    def backward_pass(self, grad_out):
        """Performs back propagation through affine, batch norm, and ReLU layers

        Args:
            grad_out: Upstream gradient

        Returns:
            grad_x: Gradient w.r.t. input
            grad_w: Gradient w.r.t. weight
            grad_b: Gradient w.r.t. bias
            grad_gamma: Gradient w.r.t. gamma constant
            grad_beta: Gradient w.r.t. beta constant
        """
        grad_relu = self.relu_layer.backward_pass(grad_out)
        grad_batch_norm, grad_gamma, grad_beta = self.batch_norm_layer.backward_pass(grad_relu)
        grad_x, grad_w, grad_b = self.affine_layer.backward_pass(grad_batch_norm)

        return grad_x, grad_w, grad_b, np.sum(grad_gamma), np.sum(grad_beta)
