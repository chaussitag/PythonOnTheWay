#!/usr/bin/env python
# coding=utf8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# load the mnist data from http://yann.lecun.com/exdb/mnist/ into variable 'mnist'
# mnist.train.images:
#    contains all training images, it has a shape of [55000, 784],
#    The first dimension indexes the images,
#    and the second dimension indexes the pixels in each image of size 28 x 28 flatted into 1-d array.
# mnist.train.labels:
#    contains labels for all training images, it has a shape of [55000, 10],
#    each row is a one-hot vector, for example,
#    a image containing '3' would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_truth = tf.placeholder(tf.float32, [None, 10])

# using cross entropy loss
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_truth * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_loss)

# using L2 norm as loss
# diff = y_truth - y
# L2_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(diff * diff, reduction_indices = [1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(L2_loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# run the training
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_truth: batch_ys})

# test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_truth: mnist.test.labels})
print("simple soft-max accuracy: %f" % result)
