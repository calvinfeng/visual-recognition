import numpy as np
from conv_net.fc_network_model import FCNetworkModel


if __name__ == "__main__":
    N = 10
    num_classes = 10
    hidden_dims = [10, 10, 10, 10, 10]

    np.random.seed(1)
    rand_inputs = 100 * np.random.randn(N, 5)
    rand_labels = np.random.randint(num_classes, size=(N,))

    model = FCNetworkModel(hidden_dims, input_dim=5, weight_scale=1e-1)
    scores =  model.loss(rand_inputs)
    loss, grads = model.loss(rand_inputs, rand_labels)
    print "Loss: %s and Grads has keys %s" % (loss, str(sorted(grads.keys())))
    model.gradient_check(rand_inputs, rand_labels)

    print "Scores: %s" % str(scores.shape)
