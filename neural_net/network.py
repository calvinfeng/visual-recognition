from past.builtins import xrange

import numpy as np


class NeuralNetwork(object):
    """
    A three-layer fully-connected neural network with the following architecture:
    input -> fully-connected -> ReLU -> fully-connected -> ReLU -> fully-connected -> softmax -> prediction
    """
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = dict()
        # Input layer
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        # First hidden layer
        self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        # Second hidden layer
        self.params['W3'] = std * np.random.rand(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

    def forward_prop(self, X):
        """
        Params:
        - X: Input matrix, each row represents an input vector per example
        - N: Number of input examples
        - D: Dimension of the input vector (a.k.a input_size)
        - H: Dimension of hidden vector (a.k.a hidden_size)
        - O: Dimension of output vector (a.k.a output_size)

        Returns:
        - probs: Probabilities for each class
        """
        N, D = X.shape

        # Extracting parameters, a.k.a weights
        W1, b1 = self.params['W1'], self.params['b1'] # (D x H) + (D x H) *broadcasting technique vertically
        W2, b2 = self.params['W2'], self.params['b2'] # (H x H) + (H x H)
        W3, b3 = self.params['W3'], self.params['b3'] # (H x H) + (H x O)

        theta1 = X.dot(W1) + b1 # Multiply gate (N x D) (D x H) => (N x H)
        a1 = np.maximum(theta1, 0) # ReLU gate

        theta2 = a1.dot(W2) + b2 # Multiply gate (N x H) (H x H) => (N x H)
        a2 = np.maximum(theta2, 0) # ReLU gate

        scores = a2.dot(W3) + b3 # Multiply gate (N x H)(H x O) => (N x O)
        exp_scores = np.exp(scores) # Softmax

        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # Softmax => (N x O)

        return probs

    def predict(self, X):
        probs = self.forward_prop(X)
        return np.argmax(probs, axis=1)
