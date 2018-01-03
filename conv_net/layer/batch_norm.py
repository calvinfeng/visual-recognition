import numpy as np


class BatchNormLayer(object):
    def __init__(self):
        self.x = None
        self.norm_x = None
        self.beta = None
        self.gamma = None
        self.eps = None
        self.mean = None
        self.var = None

    def forward_pass(self, x, gamma, beta, bn_param):
        """
        Args:
            x: Input matrix of shape (N, D)
            gamma: Scale parameter of shape (D,)
            beta: Shift parameter of shape (D,)
            bn_param: Dictionary with the following keys:
                - mode: 'train' or 'test'; required
                - eps: constant for numeric stability
                - momentum: constant for running mean/variance
                - running_mean: array of shape (D,)
                - running_var: array of shape (D,)

        Returns:
            out: The output of batch normalization
        """
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.eps = bn_param.get('eps', 1e-5)

        N, D = x.shape
        momentum = bn_param.get('momentum', 0.9)
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

        mode = bn_param['mode']
        if mode == 'train':
            self.mean = x.mean(axis=0)
            self.var = x.var(axis=0)
            self.norm_x = (x - self.mean) / np.sqrt(self.var + self.eps)
            out = self.norm_x * self.gamma + self.beta

            # Formula for exponential moving average
            running_mean = momentum * running_mean + (1 - momentum) * self.mean
            running_var = momentum * running_var + (1 - momentum) * self.var

        elif mode == 'test':
            # If it is in test mode, there is no need to store the x
            norm_test_x = (x - running_mean) / np.sqrt(running_var + eps)
            out = norm_test_x * gamma + beta

        else:
            raise ValueError("Invalid forward batch normalization mode: %s" % mode)

        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var

        return out

    def backward_pass(self, grad_out, simplified=True):
        """
        Args:
            grad_out: Upstream gradient
            simplified: If simplified, the gradient computation will use a simplified implementation
        Returns:
            grad_x: Gradient with respect to x, of shape (N, D)
            grad_gamma: Gradient with respect to gamma, of shape (D,)
            grad_beta: Gradient with respect to beta, of shape (D,)
        """
        if self.x is not None and self.norm_x is not None:
            N = self.x.shape[0]
            grad_norm_x = grad_out * self.gamma

            if simplified:
                # Simplified the bottom expression
                grad_x = ((1.0 / N) * (1.0 / np.sqrt(self.var + self.eps))
                            * (N * grad_norm_x - np.sum(grad_norm_x, axis=0) - self.norm_x * np.sum(grad_norm_x * self.norm_x, axis=0)))
            else:
                # This is the result of using chain rule
                grad_var = (-0.5) * (self.x - self.mean) * (self.var + self.eps) ** (-3.0/2)
                grad_var = np.sum(grad_norm_x * grad_var, axis=0)

                grad_mean = (-1.0) / np.sqrt(self.var + self.eps)
                grad_mean = (np.sum(grad_norm_x * grad_mean, axis=0)
                                + np.sum(-2 * (self.x - self.mean) * grad_var, axis=0) * (1.0 / N))

                grad_x = 1.0 / np.sqrt(self.var + self.eps)
                grad_x = (grad_norm_x * grad_x) + (2 * grad_var * (self.x - self.mean) / N) + (grad_mean / N)

            grad_gamma = (grad_out * self.norm_x).sum(axis=0)
            grad_beta = grad_out.sum(axis=0)

        return grad_x, grad_gamma, grad_beta