from conv_net.solver import Solver
from conv_net.fc_network_model import FCNetworkModel
from conv_net.conv_network_model import ConvNetworkModel
from conv_net.data_utils import get_preprocessed_CIFAR10
import numpy as np

def main():
    # One may run a gradient check before executing the model on full training data set
    N = 20
    num_classes = 10
    hidden_dims = [20, 20]

    np.random.seed(1)
    rand_inputs = 100 * np.random.randn(N, 20)
    rand_labels = np.random.randint(num_classes, size=(N,))

    model = FCNetworkModel(hidden_dims, input_dim=20, weight_scale=1e-2, use_batchnorm=True, reg=1, update_rule='rmsprop')
    model.gradient_check(rand_inputs, rand_labels)

    ##################################
    # Fully-connected Neural Network #
    ##################################
    feed_dict = get_preprocessed_CIFAR10('../datasets/cifar-10-batches-py')

    for key, value in feed_dict.iteritems():
        print "{0:s} has shape: {1:s}".format(key, value.shape)

    # This particular fully connected model can achieve up to 52% accuracy on validation
    model = FCNetworkModel([200, 200, 200, 200, 200], weight_scale=5e-2, use_batchnorm=True, reg=0)

    solver = Solver(model,
                    feed_dict,
                    update_rule='sgd_momentum',
                    num_epochs=4,
                    batch_size=100,
                    optim_config={'learning_rate': 8e-3},
                    verbose=True)
    solver.train()

    ################################
    # Convolutional Neural Network #
    ################################
    model = ConvNetworkModel()

    t0 = time.time()

    solver = Solver(model, feed_dict,
                    update_rule='sgd_momentum',
                    num_epochs=4,
                    batch_size=100,
                    optim_config={'learning_rate': 1e-3},
                    verbose=True)
    solver.train()

    t1 = time.time()
    print "Completed training on conv. net with elapsed time: " + str(t1 - t0)


if __name__ == "__main__":
    main()
