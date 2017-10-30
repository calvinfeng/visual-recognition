from past.builtins import xrange

import numpy as np


class NeuralNetwork(object):
    """A three-layer fully-connected neural network.
    Architecture:
        input -> fully-connected -> ReLU -> fully-connected -> ReLU -> fully-connected -> softmax -> prediction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        self.params = dict()
        # Input layer
        self.params['W1'] = std * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        # First hidden layer
        self.params['W2'] = std * np.random.randn(hidden_dim, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)

        # Second hidden layer
        self.params['W3'] = std * np.random.rand(hidden_dim, output_dim)
        self.params['b3'] = np.zeros(output_dim)

    def train(self, X, y, reg=0):
        act = self._forward_prop(X)
        loss = self._loss(X, y, act['probs'], reg)
        grads = self._gradients(X, y, act)

        return True

    def predict(self, X):
        act = self._forward_prop(X)
        return np.argmax(act['probs'], axis=1)

    def _loss(self, X, y, probs, reg=0):
        """
        Args:
            X: Input matrix, each row represents an input vector for each example
            y: Label matrix, each row represents an classification vector for each example
            probs: Probabilities of classification for each example
            reg: Regularization strength

        Returns:
            loss: The total loss of the current model
        """
        loss = 0
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        for ith_example, k in np.ndenumerate(y):
            loss += -np.log(probs[ith_example][k])

        N, _ = X.shape
        loss = loss / N
        loss += reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        return loss

    def _gradients(self, X, y, act, reg=0):
        """Compute the gradients for all of the parameters within the network

        Args:
            X: Input matrix, each row represents an input vector for each example
            y: Label matrix, each row represents an classification vector for each example
            act: Activation map which contains all the activation vectors for each layer of the network
            reg: Regularization strength
        """
        N, _ = X.shape
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        # Define a gradient dictionary
        grad = dict()

        # Computing gradients of softmax score
        dscores = act['probs']
        dscores[range(N), y] -= 1
        dscores /= N # (N x O)

        # Using ReLU activation to compute gradient of W3 and b3 w.r.t. loss
        a2 = act['a2'] # Dimension: (N x H)
        grad['W3'] = np.dot(a2.T, dscores) # Dimension: (H x N)(N x O) => (H x O)
        grad['W3'] += reg*W3
        grad['b3'] = np.sum(dscores, axis=0) # Dimension: (1 x O)

        # Computing gradient of theta2 score
        da2 = np.dot(dscores, W3.T) # Dimension: (N x O)(O x H) => (N x H)
        dtheta2 = da2
        dtheta2[a2 <= 0] = 0 # Dimension: (N x H)

        # Using ReLU activation to compute gradient of W2 and b2 w.r.t loss
        a1 = act['a1'] # Dimension: (N x H)
        grad['W2'] = np.dot(a1.T, dtheta2) # Dimension: (H x N)(N x H) => (H x H)
        grad['W2'] += reg*W2
        grad['b2'] = np.sum(dtheta2, axis=0) # Dimension: (1 x H)

        # Computing gradient of theta1 score
        da1 = np.dot(dtheta2, W2.T) # Dimension: (N x H)(H x H) => (N x H)
        dtheta1 = da1
        dtheta1[a1 <= 0] = 0 # Dimension: (N x H)

        # Using ReLU activation to compute gradient of W1 and b1 w.r.t loss
        grad['W1'] = np.dot(X.T, dtheta1) # Dimension: (D x N)(N x H) = (D x H)
        grad['W1'] += reg*W1
        grad['b1'] = np.sum(dtheta1, axis=0) # Dimension: (1 x H)

        return grad

    def _forward_prop(self, X):
        """
        Args:
            X: Input matrix, each row represents an input vector for each example
            N: Number of input examples
            D: Dimension of the input vector (a.k.a input_dim)
            H: Dimension of hidden vector (a.k.a hidden_dim)
            O: Dimension of output vector (a.k.a output_dim)

        Returns:
            probs: Probabilities of classification for each example
        """
        N, D = X.shape

        # Extracting parameters, a.k.a weights
        W1, b1 = self.params['W1'], self.params['b1'] # (D x H) + D * (1 x H) *broadcasting technique vertically
        W2, b2 = self.params['W2'], self.params['b2'] # (H x H) + H * (1 x H)
        W3, b3 = self.params['W3'], self.params['b3'] # (H x O) + H * (1 x O)

        # Activations
        act = dict()

        act['theta1'] = X.dot(W1) + b1 # Multiply gate (N x D) (D x H) => (N x H)
        act['a1'] = np.maximum(act['theta1'], 0) # ReLU gate

        act['theta2'] = act['a1'].dot(W2) + b2 # Multiply gate (N x H) (H x H) => (N x H)
        act['a2'] = np.maximum(act['theta2'], 0) # ReLU gate

        act['scores'] = act['a2'].dot(W3) + b3 # Multiply gate (N x H)(H x O) => (N x O)
        act['exp_scores'] = np.exp(act['scores']) # Softmax

        act['probs'] = act['exp_scores'] / np.sum(act['exp_scores'], axis=1, keepdims=True) # Softmax => (N x O)

        return act


if __name__ == "__main__":
    from neural_net.tests.test_forward_prop import generate_random_data

    N = 10
    input_dim, hidden_dim, output_dim = 5, 5, 5
    rand_X, rand_y = generate_random_data(N, input_dim, output_dim)
    network = NeuralNetwork(input_dim, hidden_dim, output_dim, std=0.25)

    network.train(rand_X, rand_y)
