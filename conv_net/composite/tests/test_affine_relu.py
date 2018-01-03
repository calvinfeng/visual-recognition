from conv_net.composite.affine_relu import AffineReLU
from conv_net.gradient_check import *
import numpy as np
import unittest


class AffineReLUTest(unittest.TestCase):
    def setUp(self):
        self.layer = AffineReLU()

    def test_backward_pass(self):
        np.random.seed(231)
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 10)
        b = np.random.randn(10)
        dout = np.random.randn(2, 10)

        num_dx = eval_numerical_gradient_array(lambda x: self.layer.forward_pass(x, w, b), x, dout)
        num_dw = eval_numerical_gradient_array(lambda w: self.layer.forward_pass(x, w, b), w, dout)
        num_db = eval_numerical_gradient_array(lambda b: self.layer.forward_pass(x, w, b), b, dout)

        dx, dw, db = self.layer.backward_pass(dout)

        self.assertAlmostEqual(rel_error(num_dx, dx), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_dw, dw), 1e-9, places=2)
        self.assertAlmostEqual(rel_error(num_db, db), 1e-9, places=2)
