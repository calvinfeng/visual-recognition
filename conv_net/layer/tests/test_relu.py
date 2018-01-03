from conv_net.layer.relu import ReLU
from conv_net.gradient_check import *
import numpy as np
import unittest


class ReluGateTest(unittest.TestCase):
    def setUp(self):
        self.layer = ReLU()

    def test_forward_pass(self):
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        output = self.layer.forward_pass(x)
        expected_output = np.array([[ 0.,          0.,          0.,          0.,        ],
                                    [ 0.,          0.,          0.04545455,  0.13636364,],
                                    [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

        self.assertAlmostEqual(rel_error(output, expected_output), 1e-9, places=2)

    def test_backward_pass(self):
        np.random.seed(231)
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        dx_num = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x), x, dout)
        dx = self.layer.backward_pass(dout)

        self.assertAlmostEqual(rel_error(dx_num, dx), 1e-9, places=2)
