import numpy as np


class Solver(object):
    """A solver encapsulates all the logic necessary for training classification models. The solver performs stochastic
    gradient descent using different update rules defined in optimizer.
    """
    def __init__(self, model, data, **kwargs):
        """
        Required args:
            model:
            data:

        Optional args:
            update_rule: A string giving the name of an update rule in optimizer.py, default is sgd
            optim_config: A dictionary containing hyperparameters that will be passed to the choosen update rule.
            lr_decay:
            batch_size:
            num_epochs:
            print_every:
            verbose:
            num_train_samples:
            num_val_samples:
            checkpoint_name:
        """

    def _reset(self):

    def _step(self):

    def _save_checkpoint(self):

    def check_accuracy(self, x, y, num_samples=None, batch_size=100):

    def train(self):
