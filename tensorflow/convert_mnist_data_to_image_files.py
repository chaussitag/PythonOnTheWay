#!/usr/bin/env python
# coding=utf8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os

## load the mnist data from http://yann.lecun.com/exdb/mnist/ into variable 'mnist'
## mnist.train.images:
##    contains all training images, it has a shape of [55000, 784],
##    The first dimension indexes the images,
##    and the second dimension indexes the pixels in each image of size 28 x 28 flatted into 1-d array.
## mnist.train.labels:
##    contains labels for all training images, it has a shape of [55000, 10],
##    each row is a one-hot vector, for example,
##    a image containing '3' would be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
mnist = input_data.read_data_sets("MNIST_data")
raw_data = tf.placeholder(tf.uint8, [28, 28, 3])
scaled_data = tf.image.resize_image_with_crop_or_pad(raw_data, 239, 239)
encode_jpg_tensor = tf.image.encode_jpeg(scaled_data, format = "rgb")

def save_as_jpg(data_set, dir_name, data_tag):
    global raw_data
    cnt = 0
    total = len(data_set.images)
    for i in xrange(0, total):
        label = data_set.labels[i]
        target_dir = dir_name + "/" + str(label) + "/"
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        raw_image_data = data_set.images[i].reshape(28, 28) * 255
        # original image is grayscale, convert it to rgb
        extend_raw_image_data = np.empty(shape = (28, 28, 3), dtype = raw_image_data.dtype)
        extend_raw_image_data[:, :, 0] = raw_image_data
        extend_raw_image_data[:, :, 1] = raw_image_data
        extend_raw_image_data[:, :, 2] = raw_image_data
        jpeg_data = sess.run(encode_jpg_tensor, feed_dict = {raw_data : extend_raw_image_data})
        target_file = target_dir + str(label) + "_" + str(cnt) + "_" + data_tag + ".jpg"
        with open(target_file, "w") as f:
            f.write(jpeg_data)
            cnt += 1

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    save_as_jpg(mnist.train, "MNIST_image/train", "train")
    save_as_jpg(mnist.test, "MNIST_image/test", "test")