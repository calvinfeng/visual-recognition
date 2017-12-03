import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from demo_net.data_util import load_iris_data


class SoftmaxLossNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        self.params = dict()
        self.params['W1'] = std * randn(input_dim, hidden_dim) # random normal distributed
        self.params['W2'] = std * randn(hidden_dim, output_dim) # random normal distributed

    def _forward_prop(self, x):
        acts = dict()
        
        # (N x input_dim)(input_dim x hidden_dim)
        acts['theta1'] = np.dot(x, self.params['W1']) 
        
        # (N x hidden_dim)
        acts['a1'] = 1 / (1 + np.exp(-acts['theta1']))
        
        # (N x hidden_dim)(hidden_dim x output_dim)
        acts['theta2'] = np.dot(acts['a1'], self.params['W2'])
        
        # Exponential scores can become very big numbers, in order to ensure numerical stability, we need to apply the
        # following mathematical trick (shifting by max exponential score.)
        acts['a2'] = np.exp(acts['theta2']) 
        acts['a2'] -= np.max(acts['a2'])
        acts['a2'] = acts['a2'] / np.sum(acts['a2'], axis=1, keepdims=True)

        return acts
    
    def _loss(self, acts, y):
        loss = 0
        for idx, val in enumerate(y):
            # val is the classification 0, 1, 2 etc...
            # idx is the ith example we are classifying
            y_pred = acts['a2'][idx][int(val)] 
            loss += -1 * np.log(max(y_pred, 1e-15)) # Prevent taking log(0) which gives infinity
            
        return loss / len(y)

if __name__ == "__main__":
    xtr, ytr = load_iris_data('./datasets/iris_train.csv', multi_dimen_labels=False)
    xte, yte = load_iris_data('./datasets/iris_test.csv',  multi_dimen_labels=False)
    input_dim, hidden_dim, output_dim = xtr.shape[1], 5, 3
    network = SoftmaxLossNetwork(input_dim, hidden_dim, output_dim)
    acts = network._forward_prop(xtr)
    print network._loss(acts, ytr)
    