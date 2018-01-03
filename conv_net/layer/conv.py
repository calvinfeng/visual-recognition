import numpy as np


class Conv(object):
    """Conv implements a network layer that performs convolution operation on input data

    Convolution expects an input consisting of N data points, each with C channels, height H and width W. It applies
    F different filters, where each filter spans all C channels and has height Hf and width Wf.
    """

    def __init__(self):
        self.x = None
        self.w = None
        self.b = None
        self.stride = None
        self.pad = None

    def forward_pass(self, x, w, b, stride, pad):
        """Naive implementation of forward pass for a convolutional layer, i.e. it has poor performance as compared to
        the native C implementation

        Args:
            x: Input data, of shape (N, C, H, W)
            w: Filter weights, of shape (F, C, Hf, Wf)
            b: Biases, of shape (F,)
            stride: The number of pixels between adjacent receptive fields in the horizontal and vertical directions
            pad: The number of pixels that will be used to zero-pad the input

        Returns:
            out: Output data, of shape (N, F, H_out, W_out)
        """
        N, _, H, W = x.shape
        F, _, Hf, Wf = w.shape

        # API for pad_width is ((before_1, after_1), ..., (before_N, after_N)), we are only padding the image height and
        # width, that means we don't need to worry about the first 2 dimensions.
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad))
        padded_x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

        H_out = int(1 + (H + 2 * pad - Hf) / stride)
        W_out = int(1 + (W + 2 * pad - Wf) / stride)
        out = np.zeros((N, F, H_out, W_out))

        # Now we iterate through every coordinate of the output and perform convolution on blocks of the padded input
        for n in range(N):
            for f in range(F):
                for h_out in range(H_out):
                    h_in = h_out * stride
                    for w_out in range(W_out):
                        w_in = w_out * stride
                        conv_sum = np.sum(padded_x[n][:, h_in:h_in + Hf, w_in:w_in + Wf] * w[f])
                        out[n, f, h_out, w_out] += conv_sum + b[f]

        return out
