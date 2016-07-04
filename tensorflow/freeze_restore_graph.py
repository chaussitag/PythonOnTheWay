#!/usr/bin/env python
# coding=utf8

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.client import graph_util

def train_and_freeze_graph():
    #####################################################################################
    # train a LMS model
    # generate 100 samples with Y = W * X + b, where X, Y are 2-D column vectors,
    # and W and b have following values:
    #      / 1.0  0.0 \        / 1.0 \
    # W =  |          | ,  b = |     |
    #      \ 0.0, 2.0 /        \ 0.5 /
    W_real = np.array([[1.0, 0.0], [0.0, 2.0]])
    b_real = np.array([[1.0], [0.5]])
    X_data = np.random.rand(2, 100).astype(np.float32)
    Y_data = np.dot(W_real, X_data) + b_real

    # Try to find values for W and b that compute y_data = W * x_data + b
    W = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name = "W_learn")
    b = tf.Variable(tf.zeros([2, 1]), name = "b_learn")
    x = tf.placeholder(tf.float32, [2, None], name = "x_input")
    y = tf.add(tf.matmul(W, x), b, name = "y_guess")

    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - Y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)

    grads = optimizer.compute_gradients(loss)
    apply_grad_op = optimizer.apply_gradients(grads)

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.initialize_all_variables()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    for i in xrange(200):
        _, loss_value = sess.run([apply_grad_op, loss], feed_dict={x: X_data})
        print("loss %f" % loss_value)

        ## gradients has the same shape as it's corresponding variable
        if i % 50 == 0:
            print("=======================================")
            print("type of grads: %s, size %d" % (type(grads), len(grads)))
            print("type of grad : %s, type of var %s" % (type(grads[0][0]), type(grads[0][1])))
            for grad, var in grads:
                print("gradients of %s:" % var.name)
                print(sess.run(grad, feed_dict={x: X_data}))
            print("=======================================")

    W_value, b_value = sess.run([W, b])
    print("after training:")
    print("W is:")
    print(W_value)
    print("b is:")
    print(b_value)

    print("x.name %s" % x.name)
    print("y.name %s" % y.name)
    #####################################################################################

    #####################################################################################
    # freeze and save the trained model for later restoring
    # Write out the trained graph and labels with the weights stored as constants.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), ["x_input", "y_guess"])
    with gfile.FastGFile('draft_graph.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    #####################################################################################


def create_graph():
    with tf.gfile.FastGFile('draft_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def restore_freezed_graph():
    create_graph()
    print("all variable number: %d" % len(tf.all_variables()))
    with tf.Session() as sess:
        y_tensor = sess.graph.get_tensor_by_name("y_guess:0")
        x_input = np.array([[1.0], [1.0]])
        print(x_input)
        test_result = sess.run(y_tensor, {"x_input:0": x_input})
        print(test_result)

if __name__ == "__main__":
    train_and_freeze_graph()
    # restore_freezed_graph()
