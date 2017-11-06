import numpy as np
from numpy.random import randn
from demonet.data_util import load_iris_data

class SimpleNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        self.params = dict()
        self.params['W1'] = std * randn(input_dim, hidden_dim) # random normal distributed
        self.params['W2'] = std * randn(hidden_dim, output_dim) # random normal distributed

    def train(self, x, y, learning_rate=1e-2, num_iters=2000):
        loss_hist = []
        for it in xrange(num_iters):
            acts = self._forward_prop(x)
            loss = self._loss(acts, y)
            grads = self._backward_prop(x, y, acts)

            loss_hist.append(loss)

            self.params['W2'] -= learning_rate * grads['W2']
            self.params['W1'] -= learning_rate * grads['W1']

        return loss_hist

    def predict(self, x):
        act = self._forward_prop(x)
        output_dim = act['a2'].shape[1]
        return np.argmax(act['a2'], axis=1)

    def _forward_prop(self, x):
        acts = dict()
        acts['theta1'] = np.dot(x, self.params['W1']) # (N x input_dim)(input_dim x hidden_dim)
        acts['a1'] = 1 / (1 + np.exp(-acts['theta1'])) # (N x hidden_dim)
        acts['theta2'] = np.dot(acts['a1'], self.params['W2']) # (N x hidden_dim)(hidden_dim x output_dim)
        acts['a2'] = 1 / (1 + np.exp(-acts['theta2'])) # known as scores (N x output_dim)
        return acts

    def _loss(self, acts, y):
        accum = np.square(acts['a2'] - y).sum()
        N = acts['a2'].shape[0]
        return accum / N

    def _backward_prop(self, x, y, acts):
        grads = dict()
        grads['a2'] = 2.0 * (acts['a2'] - y) # (N x output_dim)
        grads['theta2'] = grads['a2'] * ((1 - acts['a2']) * acts['a2']) # (N x output_dim)
        grads['W2'] = np.dot(acts['a1'].T, grads['theta2']) # (hidden_dim x N)(N x output_dim)
        grads['a1'] = np.dot(grads['theta2'], self.params['W2'].T) # (N x output_dim)(output_dim x hidden_dim)
        grads['theta1'] = grads['a1'] * ((1 - acts['a1']) * acts['a1']) # (N x hidden_dim)
        grads['W1'] = np.dot(x.T, grads['theta1']) # (input_dim x N)(N x hidden_dim)
        return grads


if __name__ == "__main__":
    xtr, ytr = load_iris_data('./iris_train.csv')
    xte, yte = load_iris_data('./iris_test.csv')
    input_dim, hidden_dim, output_dim = xtr.shape[1], 5, ytr.shape[1]
    network = SimpleNetwork(input_dim, hidden_dim, output_dim)

    test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
    print 'Test accuracy before training: %s' % str(test_acc)

    network.train(xtr, ytr)

    test_acc = (network.predict(xte) == np.argmax(yte, axis=1)).mean()
    print 'Test accuracy after training: %s' % str(test_acc)
