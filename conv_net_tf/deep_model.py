class DeepModel(object):
    """A deep model

    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    - 7x7 convolutional layer with 32 filters, stride = 1
    - ReLU
    - 2x2 max pooling, stride = 2
    - Spatial batch normalization
    - Do above 3 times
    - Flatten it to Nx1024
    - Dense ReLU
    - Dropout 50%
    - Dense affine to Nx10
    """
    def __init__(self):
        tf.reset_default_graph()
