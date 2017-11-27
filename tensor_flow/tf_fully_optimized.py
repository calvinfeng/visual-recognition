import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

N, D, H = 64, 1000, 100
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))

    xavier_init = tf.contrib.layers.xavier_initializer()
    h = tf.layers.dense(inputs=x, units=H, activation=tf.nn.relu, kernel_initializer=xavier_init)
    y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=xavier_init)
    loss = tf.losses.mean_squared_error(y_pred, y)

    optimizer = tf.train.GradientDescentOptimizer(7e0)
    weight_updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D)
    }

    losses, iters = [], []
    for t in range(500):
        loss_val, _ = sess.run([loss, weight_updates], feed_dict=values)
        losses.append(loss_val)
        iters.append(t)

plt.plot(iters, losses)
plt.show()
