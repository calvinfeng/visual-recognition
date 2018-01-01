from conv_net.layer.relu_gate import ReLUGate
from conv_net.gradient_check import *
import numpy as np


if __name__ == "__main__":
    # Test the affine_forward function
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
    gate = ReLUGate()

    output = gate.forward_pass(x)
    correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                            [ 0.,          0.,          0.04545455,  0.13636364,],
                            [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

    # Compare your output with ours. The error should be around 5e-8
    print "Testing ReLUGate forward pass function:"
    print "Error: %s" % rel_error(output, correct_out)

    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: gate.forward_pass(x), x, dout)

    dx = gate.backward_pass(dout)

    # The error should be around 3e-12
    print "Testing ReLUGate backward pass function:"
    print "Error: %s" % rel_error(dx_num, dx)
