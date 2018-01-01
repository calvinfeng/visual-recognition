import numpy as np


def sgd(w, dw, config=None):
    """Performs vanilla stochastic gradient descent

    Args:
        w: matrix weights
        dw: gradients of weights
        config:
            - learning_rate: scalar learning rate
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw

    return w, config
