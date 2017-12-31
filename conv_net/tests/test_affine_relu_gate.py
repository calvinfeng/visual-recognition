from conv_net.layer.affine_relu_gate import AffineReLUGate
from conv_net.gradient_check import *
import numpy as np

np.random.seed(231)
x = np.random.randn(2, 3, 4)
w = np.random.randn(12, 10)
b = np.random.randn(10)
dout = np.random.randn(2, 10)

gate = AffineReLUGate()
output = gate.forward_pass(x, w, b)
dx, dw, db = gate.backward_pass(dout)

num_dx = eval_numerical_gradient_array(lambda x: gate.forward_pass(x, w, b), x, dout)
num_dw = eval_numerical_gradient_array(lambda w: gate.forward_pass(x, w, b), w, dout)
num_db = eval_numerical_gradient_array(lambda b: gate.forward_pass(x, w, b), b, dout)

print "Testing backward pass:"
print "dx error: %s" % rel_error(num_dx, dx)
print "dw error: %s" % rel_error(num_dw, dw)
print "db error: %s" % rel_error(num_db, db)
