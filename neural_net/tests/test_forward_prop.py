import numpy as np
import unittest
from neural_net.network import NeuralNetwork

def generate_random_data(num_inputs, input_dim, num_classes):
    X = 10 * np.random.randn(num_inputs, input_dim)
    y = np.random.randint(num_classes, size=num_inputs)
    return X, y


# Run nosetests --nocapture
class ForwardPropTests(unittest.TestCase):
    def setUp(self):
        self.input_dim, self.hidden_dim, self.num_classes = 4, 10, 5
        self.num_inputs = 10
        self.network = NeuralNetwork(self.input_dim, self.hidden_dim, self.num_classes, std=0.25)
        self.rand_X, self.rand_y = generate_random_data(self.num_inputs, self.input_dim, self.num_classes)

    def test_forward_prop(self):
        # Initializing the network, using a small standard deviation, the network vanishes very quickly!
        probs = self.network.forward_prop(self.rand_X)
        print probs
        for sum in np.sum(probs, axis=1):
            self.assertAlmostEqual(sum, 1)

    def test_predict(self):
        pred_arr = self.network.predict(self.rand_X)
        print pred_arr
        for pred in pred_arr:
            self.assertTrue(0 <= pred <= self.num_classes - 1)

    def test_loss(self):
        loss = self.network.loss(self.rand_X, self.rand_y)
        print loss
        self.assertTrue(isinstance(loss, float))
