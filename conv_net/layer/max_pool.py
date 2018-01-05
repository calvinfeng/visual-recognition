import numpy as np


class MaxPool(object):
    """MaxPool implements a network layer that performs max pooling operation on input
    """
    def __init__(self):
        self.x = None
        self.pool_height = None
        self.pool_width = None
        self.stride = None

    def forward_pass(self, x, pool_height, pool_width, stride):
        """Naive implementation of forward pass for a max pooling layer, i.e. it has poor performance as compared to the
        native C implementation

        Args:
            x: Input data, of shape (N, C, H, W)
            pool_height: Height of each pooling region
            pool_width: Width of each pooling region
            stride: Distance between adjacent pooling regions

        Returns:
            out: Output data, of shape (N, C, H_out, W_out)
        """
        self.x, self.pool_height, self.pool_width, self.stride = x, pool_height, pool_width, stride
        N, C, H, W = x.shape
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)

        out = np.zeros((N, C, H_out, W_out))
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    h_in = h * stride
                    for w in range(W_out):
                        w_in = w * stride
                        out[n, c, h, w] = np.max(x[n, c, h_in:h_in + pool_height, w_in:w_in + pool_width])
        return out


    def backward_pass(self, grad_out):
        """Naive implementation of backward pass for a max pooling layer, i.e. it has poor performanced as compared to
        the native C implementation. The derivative for max pooling is similarly to that of ReLU. The max value of a
        pooling region gets 1 as its derivative, the rest receives 0.

        Args:
            grad_out: Upstream gradients

        Returns:
            grad_x: Gradient w.r.t. to input x
        """
        if self.x is not None:
            N, C, H, W = self.x.shape
            H_out = int(1 + (H - self.pool_height) / self.stride)
            W_out = int(1 + (W - self.pool_width) / self.stride)

            grad_x = np.zeros(self.x.shape)
            for n in range(N):
                for c in range(C):
                    for h in range(H_out):
                        h_in = h * self.stride
                        for w in range(W_out):
                            w_in = w * self.stride
                            curr_pool = self.x[n, c, h_in:h_in + self.pool_height, w_in:w_in + self.pool_width]
                            max_multi_idx = np.unravel_index(curr_pool.argmax(), curr_pool.shape)
                            grad_x[n, c][h_in + max_multi_idx[0], w_in + max_multi_idx[1]] += 1 * grad_out[n, c, h, w]
            return grad_x
