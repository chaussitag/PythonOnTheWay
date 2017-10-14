#!/usr/bin/env python
# coding=utf8

import argparse
import numpy as np
import cv2
import os
import sys

# a 3x3 gaussian convolution kernel converting 3 input channels to 1 output channels
gaussian_weights_3x3 = \
    np.array([[[[0.095, 0.118, 0.095], [0.118, 0.148, 0.118], [0.095, 0.118, 0.095]],
               [[0.095, 0.118, 0.095], [0.118, 0.148, 0.118], [0.095, 0.118, 0.095]],
               [[0.095, 0.118, 0.095], [0.118, 0.148, 0.118], [0.095, 0.118, 0.095]]]], dtype=np.float32)

# a 5x5 gaussian convolution kernel converting 3 input channels to 1 output channels
gaussian_weights_5x5 = (1.0 / 273.0) * \
                       np.array([[[[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]],
                                  [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]],
                                  [[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]]]],
                                dtype=np.float32)


# a wrapper class for convolution parameters
# conv_weights: each kernel in one row,
#               should be an array of 'out_channels x in_channels x kernel_h x kernel_w'
class ConvParameter(object):
    def __init__(self, stride_h=1, stride_w=1,
                 pad_h=1, pad_w=1,
                 conv_weights=gaussian_weights_3x3):
        self._stride_h = stride_h
        self._stride_w = stride_w
        self._pad_h = pad_h
        self._pad_w = pad_w

        w_shape = conv_weights.shape
        assert len(w_shape) == 4
        self._out_channels = w_shape[0]
        self._in_channels = w_shape[1]
        self._kernel_h = w_shape[2]
        self._kernel_w = w_shape[3]
        self._conv_weights = conv_weights.reshape(-1, self.in_channels * self.kernel_h * self.kernel_w)

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_h(self):
        return self._kernel_h

    @property
    def kernel_w(self):
        return self._kernel_w

    @property
    def stride_h(self):
        return self._stride_h

    @property
    def stride_h(self):
        return self._stride_h

    @property
    def stride_w(self):
        return self._stride_w

    @property
    def pad_h(self):
        return self._pad_h

    @property
    def pad_w(self):
        return self._pad_w

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def conv_weights(self):
        return self._conv_weights


# convert the input image (num_channels x h x w) to a matrix,
# so that convolution can be represented as matrix multiplication.
# input_imgs: input images of form 'num_channels x h x w'
def img2col(input_imgs, conv_param):
    input_channels, input_h, input_w = input_imgs.shape

    output_h = (input_h + 2 * conv_param.pad_h - conv_param.kernel_h) / conv_param.stride_h + 1
    output_w = (input_w + 2 * conv_param.pad_w - conv_param.kernel_w) / conv_param.stride_w + 1
    single_channel_output_size = output_h * output_w * conv_param.kernel_h * conv_param.kernel_w
    output_imgs = np.empty(input_channels * single_channel_output_size, dtype=input_imgs.dtype)
    output_index = 0
    # pick image patch of size kernel_h x kernel_w from each channel,
    # the pick order is from left to right and top to bottom
    for input_row_index in range(-conv_param.pad_h, input_h + conv_param.pad_h - conv_param.kernel_h + 1,
                                 conv_param.stride_h):
        for input_col_index in range(-conv_param.pad_w, input_w + conv_param.pad_w - conv_param.kernel_w + 1,
                                     conv_param.stride_w):
            for channel in range(0, input_channels):
                for y in range(input_row_index, input_row_index + conv_param.kernel_h):
                    for x in range(input_col_index, input_col_index + conv_param.kernel_w):
                        if 0 <= y < input_h and 0 <= x < input_w:
                            output_imgs[output_index] = input_imgs[channel][y][x]
                        else:
                            output_imgs[output_index] = 0
                        output_index += 1
    output_imgs = output_imgs.reshape((-1, input_channels * conv_param.kernel_h * conv_param.kernel_w))
    # tranpose the output matrix, so that each row is a list of patches from the same location of each channel
    return output_imgs.transpose()


# weights: convolution weights, each kernel in one row, one kernel per channel
#          should be an array of '1 x 3 x kernel_h x kernel_w'
def rgbImageConvolution(input_img, weights):
    input_channels, input_h, input_w = input_img.shape
    assert input_channels == 3, "please use rgb image to test"
    assert len(weights.shape) == 4
    assert weights.shape[0] == 1
    assert weights.shape[1] == 3
    conv_param = ConvParameter(conv_weights=weights)
    out_channels = conv_param.out_channels

    image_as_matrices = img2col(input_img, conv_param)
    print(conv_param.conv_weights.shape)
    print(image_as_matrices.shape)
    # convolution as matrix multiplication
    out_imgs = np.dot(conv_param.conv_weights, image_as_matrices.astype(conv_param.conv_weights.dtype))
    out_imgs /= 3.0
    out_imgs = out_imgs.astype(input_img.dtype)

    output_h = (input_h + 2 * conv_param.pad_h - conv_param.kernel_h) / conv_param.stride_h + 1
    out_imgs = out_imgs.reshape(out_channels, output_h, -1)
    return out_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convolution using matrix multiplication, like the way in caffe")
    parser.add_argument('--input_image', '-i', dest='input_image', help='path to some rgb image', required=True)
    args = parser.parse_args()
    test_image_path = args.input_image
    if not os.path.exists(args.input_image):
        print("the test image %s does not exist" % (args.input_image,))
        sys.exit(-1)
    img = cv2.imread(test_image_path)
    img = img.transpose(2, 0, 1)
    result_img = rgbImageConvolution(img, gaussian_weights_5x5)
    result_img = result_img.transpose(1, 2, 0)
    cv2.imshow("convolution using matrix multiplication", result_img)
    cv2.waitKey()
