from conv_net.layer.affine_relu_gate import AffineReLUGate
from conv_net.gradient_check import *
import numpy as np
import unittest


class AffineReLUGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = AffineReLUGate()

    def test_backward_pass(self):
        np.random.seed(231)
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        dout = np.random.randn(2, 10)

        num_dx = eval_numerical_gradient_array(lambda x: self.gate.forward_pass(x, w, b), x, dout)
        num_dw = eval_numerical_gradient_array(lambda w: self.gate.forward_pass(x, w, b), w, dout)
        num_db = eval_numerical_gradient_array(lambda b: self.gate.forward_pass(x, w, b), b, dout)

        dx, dw, db = self.gate.backward_pass(dout)

        self.assertAlmostEqual(rel_error(num_dx, dx), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_dw, dw), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_db, db), 1e-9, places=2)
