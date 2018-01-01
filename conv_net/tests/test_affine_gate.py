from conv_net.layer.affine_gate import AffineGate
from conv_net.gradient_check import *
import numpy as np

# TODO: Rewrite this using unittest
# Test the affine_forward function
num_inputs = 2
input_shape = (4, 5, 6)
output_dim = 3

input_size = num_inputs * np.prod(input_shape)
weight_size = output_dim * np.prod(input_shape)

x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
b = np.linspace(-0.3, 0.1, num=output_dim)

gate = AffineGate()

output = gate.forward_pass(x, w, b)
correct_output = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])

print "Testing AffineGate forward pass function:"
print "forward pass error: %s" % rel_error(output, correct_output)

# Test the affine_backward function
np.random.seed(231)
x = np.random.randn(10, 2, 3)
w = np.random.randn(6, 5)
b = np.random.randn(5)
dout = np.random.randn(10, 5)

num_dx = eval_numerical_gradient_array(lambda x: gate.forward_pass(x, w, b), x, dout)
num_dw = eval_numerical_gradient_array(lambda w: gate.forward_pass(x, w, b), w, dout)
num_db = eval_numerical_gradient_array(lambda b: gate.forward_pass(x, w, b), b, dout)

dx, dw, db = gate.backward_pass(dout)

print "Testing AffineGate backward pass function:"
print "dx error: %s" % rel_error(num_dx, dx)
print "dw error: %s" % rel_error(num_dw, dw)
print "db error: %s" % rel_error(num_db, db)
