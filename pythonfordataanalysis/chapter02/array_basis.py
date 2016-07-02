#!/usr/bin/env python
#coding=utf8

import numpy as np

# numpy.ndarray has a dtype specify the data type of all elements

def array_creation():
    # simple creation
    arr = np.array([1.0, 2.0])
    print(arr.shape) # (2L,0)

    arr = np.array([[1.0], [2.0]])
    print(arr.shape) # (2L, 1L)

    # initialize with another ndarray, data copied
    arr1 = np.array(arr)
    arr1[0][0] = -1.0
    print(arr[0][0] == arr1[0][0]) # False

    # helper functions to create some kinds of ndarray
    arr = np.arange(5)
    print(arr) # [0 1 2 3 4]

    arr = np.ones(3) # [1.0, 1.0, 1.0]
    print(arr)

    # [[1.  1.  1.]
    #  [1.  1.  1.]]
    arr = np.ones((2, 3))
    print(arr)

    arr = np.zeros((2, 3))
    print(arr)

    arr = np.empty((2, 3, 2))
    print(arr)

    arr = np.ones_like(arr)
    print(arr)

    arr = np.identity(4)
    print(arr)

    # [[1.  0.  0.  0.]
    #  [0.  1.  0.  0.]
    #  [0.  0.  1.  0.]]
    arr = np.eye(3, 4)
    print(arr)

def array_data_types():
    arr = np.array([1, 2, 3])
    print(arr.dtype) # int32

    arr1 = np.array(arr, dtype = np.float32)
    print(arr1.dtype) # float32

    arr = np.array(["hello", 1, 2])
    print(arr.dtype) # |S5

    arr = np.array([1, 2, 3, 4, 5], dtype = np.int64)
    print(arr.dtype) # int64
    # data copied
    float_arr = arr.astype(np.float64)
    print(float_arr.dtype) # float64

    # convert string number to float type
    numeric_string = np.array(["1.24", "-3.2", "42"])
    print(numeric_string.dtype) # |S3
    print(numeric_string) # ['1.24' '-3.2' '42']
    converted_number = numeric_string.astype(np.float64)
    print(converted_number.dtype) # float64
    print(converted_number) # [  1.24  -3.2   42.  ]

def array_index_and_sclice():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # arr[1][2] is the same as arr[1, 2]
    print(arr[1][2]) # 6
    print(arr[1, 2]) # 6

    sclice = arr[:2, 1:]
    # [[2 3]
    #  [5 6]]
    print(sclice)
    # sclice don't copy data, elements are shared with original array
    sclice[:, :] = -1
    # [[-1 - 1]
    #  [-1 - 1]]
    print(sclice)
    # [[1 - 1 - 1]
    #  [4 - 1 - 1]
    #  [7  8  9]]
    print(arr)

def boolean_index():
    names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
    # element-wise compare, generate a boolean array with the same shape
    print(names == "Bob") # [ True False False True False False False]
    print((names == "Bob") | (names == "Will")) # [ True False  True  True  True False False]

    data = np.random.randn(7, 4)
    print(data)
    print("")
    # names == "Bob" <====> [ True False False True False False False]
    # data[names == "Bob"] select the first and fourth row
    print(data[names == "Bob"])

    print("")
    # index with boolean index and sclice
    print(data[names == "Bob", :2])

if __name__ == "__main__":
    array_creation()
    print("===============================")
    array_data_types()
    print("===============================")
    array_index_and_sclice()
    print("===============================")
    boolean_index()
