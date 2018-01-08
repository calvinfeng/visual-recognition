from composite import *
from layer import *
from gradient_check import eval_numerical_gradient, rel_error
import numpy as np


class ConvNetworkModel(object):
    """ConvNetworkModel implements a convolutional network with the following architecture

    conv -> relu -> 2x2 max pool -> batch norm -> affine -> relu -> affine -> softmax

    Assuming input dimension is of the format NCHW, TODO: change data format to NHWC
    """
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, hidden_dim=100, num_classes=10,
                weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.reg = reg
        self.dtype = dtype
        self.params = dict()
        # Manually define weights
        # Input starts with (N, 3, 32, 32)
        #   1. Convolutional layer will produce (N, 32, 32, 32) using 32 7x7 filters with stride = 1
        #   2. ReLu layer will conserve dimension
        #   3. Max pooling layer will produce (N, 32, 16, 16) using 2x2 filter with stride = 2
        #   4. Batch normalization layer will conserve dimension
        #   5. Affine layer will produce (N, hidden_dim)
        #   6. ReLU layer will conserve dimension
        #   7. Affine layer will produce (N, num_classes)
        F, C, H, W = (num_filters,) + input_dim
        self.params['W1'] = np.random.normal(0, scale=weight_scale, size=(F, C, filter_size, filter_size))
        self.params['b1'] = np.zeros((F,))

        self.params['W2'] = np.random.normal(0, scale=weight_scale, size=(F * (H // 2) * (W // 2), hidden_dim))
        self.params['b2'] = np.zeros((hidden_dim,))

        self.params['W3'] = np.random.normal(0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros((num_classes,))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


if __name__ == "__main__":
    model = ConvNetworkModel()
    print model.params
