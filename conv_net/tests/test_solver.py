from conv_net.solver import Solver
from conv_net.neural_net_model import NeuralNetModel
import numpy as np


if __name__ == "__main__":
    N = 10
    num_classes = 10
    hidden_dims = [10, 10, 10, 10]

    np.random.seed(1)
    rand_inputs = 100 * np.random.randn(N, 5)
    rand_labels = np.random.randint(num_classes, size=(N,))

    model = NeuralNetModel(hidden_dims, input_dim=5)

    data = {
        'x_train': [],
        'y_train': [],
        'x_val': [],
        'y_val': []
    }

    solver = Solver(model, data)
