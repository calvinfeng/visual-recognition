"""
Minimal character-level Vanilla RNN model, took from Andrej Karpathy
"""
import numpy as np


# I/O
data = open('input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

print 'Data has %d characters, %d unique.' % (data_size, vocab_size)

# Generate a dictionary mapping
char_to_ix = { ch:i for i, ch in enumerate(chars) }
ix_to_char = { i:ch for i, ch in enumerate(chars) }

# Hyperparameters
hidden_size = 100  # Size of hidden layer of neurons
seq_length = 25  # Number of steps to unroll the RNN for
learning_rate = 1e-1
weight_scale = 1e-2

# Model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * weight_scale  # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * weight_scale  # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * weight_scale  # hidden to output
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


def rnn_loss(inputs, targets, h_prev):
    """Returns the loss, gradients on model parameters, and last hidden state

    :param inputs: A list of integers
    :type inputs: list
    :param targets: A list of integers
    :type targets: list
    :param h_prev: (H, 1) array of initial hidden state
    :type h_prev: numpy.array
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)

    loss = 0
    for t in xrange(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # Encod in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t])) + np.dot(Whh, hs[t-1] + bh)  # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # Unnormalized log prob for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # Probs for next chars
        loss += np.log(ps[t][targets[t], 0])  # Softmax

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(xrange(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1  # Softmax gradient
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # Backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # Backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # Clip to mitigate exploding gradients

    return loss, dWxh, dWhy, dbh, dby, hs[len(inputs)-1]
