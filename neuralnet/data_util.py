import cPickle as pickle
import numpy as np
import platform
import os

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """Load a single batch of CIFAR-10 data
    """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y
        
def load_CIFAR10(dir):
    """Load all CIFAR-10 data
    """
    xs = []
    ys = []
    for b in range(1, 6):
        filepath = os.path.join(dir, 'data_batch_%d' % b)
        X, Y = load_CIFAR_batch(filepath)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

if __name__ == "__main__":
    x_training, y_training, x_test, y_test = load_CIFAR10('../cifar-10-batches')
    print x_training.shape
    print y_training.shape