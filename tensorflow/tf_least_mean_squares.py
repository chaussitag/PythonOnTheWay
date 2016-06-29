#!/usr/bin/env python
# coding=utf8

import tensorflow as tf
import numpy as np

# generate 100 samples with Y = W * X + b, where X, Y are 2-D column vectors,
# and W and b have following values:
#      / 1.0  0.0 \        / 1.0 \
# W =  |          | ,  b = |     |
#      \ 0.0, 2.0 /        \ 0.5 /
W_real = np.array([[1.0, 0.0], [0.0, 2.0]])
b_real = np.array([[1.0], [0.5]])
# the sample data
X_samples = np.random.rand(2, 100).astype(np.float32)
Y_samples = np.dot(W_real, X_samples) + b_real

# Try to find values for W and b that compute y_data = W * x_data + b
W_learn = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name = "W_learn")
b_learn = tf.Variable(tf.zeros([2, 1]), name = "b_learn")
Y_eval = tf.matmul(W_learn, X_samples) + b_learn

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(Y_eval - Y_samples))
optimizer = tf.train.GradientDescentOptimizer(0.5)

grads = optimizer.compute_gradients(loss)
apply_grad_op = optimizer.apply_gradients(grads)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    for i in xrange(200):
        _, loss_value = sess.run([apply_grad_op, loss])
        print("loss %f" % loss_value)

        ## gradients has the same shape as it's corresponding variable
        if i % 50 == 0:
            print("=======================================")
            print("type of grads: %s, size %d" % (type(grads), len(grads)))
            print("type of grad : %s, type of var %s" % (type(grads[0][0]), type(grads[0][1])))
            for grad, var in grads:
                print("gradients of %s:" % var.name)
                print(sess.run(grad))
            print("=======================================")

    W_value, b_value = sess.run([W_learn, b_learn])
    print("after training:")
    print("W is:")
    print(W_value)
    print("b is:")
    print(b_value)