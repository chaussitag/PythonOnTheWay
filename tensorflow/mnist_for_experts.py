#!/usr/bin/env python
# coding=utf8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x_, W):
    return tf.nn.conv2d(x_, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x_):
    return tf.nn.max_pool(x_, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

## load the data
mnist = input_data.read_data_sets("MNIST_data", one_hot = True)

x = tf.placeholder(tf.float32, shape = [None, 784])
y_truth = tf.placeholder(tf.float32, shape = [None, 10])

## reshape the input to 28 x 28 image suitable for convolution
x_image = tf.reshape(x, [-1, 28, 28, 1])

## the first convolutional layer:
##    32 5x5 convolution kernel, generating 32 28x28 feature map,
##    then using 2x2 max-pool to reduce each 28x28 feature map to 14x14
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

hidden1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
hidden1_pool = max_pool_2x2(hidden1)

## the second convolutional layer:
##    32 14x14 feature map ouput by the 1st layer,
##    64 5x5 convolution kernel, generating 64 14x14 feature map
##    then using 2x2 max-pool to reduce each generated 14x14 feature map to 7x7
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

hidden2 = tf.nn.relu(conv2d(hidden1_pool, W_conv2) + b_conv2)
hidden2_pool = max_pool_2x2(hidden2)

## fully connected layer
## flatten
hidden2_pool_flat = tf.reshape(hidden2_pool, [-1, 7 * 7 * 64])
W_full = weight_variable([7 * 7 * 64, 1024])
b_full = bias_variable([1024])

flatten = tf.nn.relu(tf.matmul(hidden2_pool_flat, W_full) + b_full)

## dropout
keep_prob = tf.placeholder(tf.float32)
flatten_drop = tf.nn.dropout(flatten, keep_prob)

## softmax
W_softmax = weight_variable([1024, 10])
b_softmax = bias_variable([10])

y = tf.nn.softmax(tf.matmul(flatten_drop, W_softmax) + b_softmax)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_truth * tf.log(y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    #if i % 100 == 0:
    #    train_accuracy = accuracy.eval(
    #        feed_dict = {x : batch[0], y_truth : batch[1], keep_prob : 1.0})
    #    print("step %d, training accuracy %g" % (i, train_accuracy))

    _, loss = sess.run([train_step, cross_entropy],
                       feed_dict = {x: batch[0], y_truth: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        print("step %d, loss %g" % (i, loss))
    #train_step.run(
    #    feed_dict = {x : batch[0], y_truth : batch[1], keep_prob : 0.5})

print("test accuracy %g" % accuracy.eval(
    feed_dict = {x : mnist.test.images[:2000], y_truth : mnist.test.labels[:2000], keep_prob : 1.0}))
