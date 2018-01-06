from conv_net.solver import Solver
from conv_net.fc_network_model import FCNetworkModel
import numpy as np
from conv_net import data_utils

if __name__ == "__main__":

    feed_dict = data_utils.get_preprocessed_CIFAR10('../datasets/cifar-10-batches-py')

    for key, value in feed_dict.iteritems():
        print "{0:s} has shape: {1:s}".format(key, value.shape)

    model = FCNetworkModel([200, 200, 200, 200, 200],
                        weight_scale=5e-2,
                        use_batchnorm=True,
                        reg=0)
    # model.gradient_check(feed_dict['X_val'][0:1], feed_dict['y_val'][0:1])
    params_to_train = []
    for key in model.params:
        params_to_train.append(key)

    print params_to_train
    solver = Solver(model, feed_dict,
                    update_rule='sgd_momentum',
                    num_epochs=4,
                    batch_size=100,
                    optim_config={'learning_rate': 8e-3},
                    verbose=True)
    solver.train()

    # 52% accuracy
