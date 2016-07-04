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
    # [[-1 -1]
    #  [-1 -1]]
    print(sclice)
    # [[1 -1 -1]
    #  [4 -1 -1]
    #  [7  8  9]]
    print(arr)

def boolean_index():
    names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
    # element-wise compare, generate a boolean array with the same shape
    print(names == "Bob") # [ True False False True False False False]

    # ndarray supports 'bitwise' operation such as &, |, but not support ||, &&
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

    print("")
    # reset all negtive elements to 0
    data[data < 0] = 0.0
    print(data)

def fancy_index():
    arr = np.empty((8, 4))
    for i in range(len(arr)):
        arr[i] = i
    print(arr)

    print("")
    # select rows by row index, data copied
    # [[4. 4.  4.  4.]
    #  [3. 3.  3.  3.]
    #  [0. 0.  0.  0.]
    #  [6. 6.  6.  6.]]
    arr1 = arr[[4, 3, 0, 6]]
    print(arr1)

def axis_transpose():
    # ndarray.transpose():
    # Returns a view of the array with axes transposed.
    # For a 1-D array, this has no effect.(To change between column and
    # row vectors, first cast the 1-D array into a matrix object.)
    #
    # For a 2-D array, this is the usual matrix transpose.
    #
    # For an n-D array, if axes are given, their order indicates how the
    # axes are permuted(see Examples).If axes are not provided and
    # ``a.shape = (i[0], i[1], ... i[n - 2], i[n - 1])``, then
    # ``a.transpose().shape = (i[n - 1], i[n - 2], ... i[1], i[0])``.
    arr = np.arange(6).reshape(2, 3)
    print(arr)
    print("")
    print(arr.transpose())

    print("")
    arr = np.arange(16).reshape(2, 2, 4)
    print(arr)
    print("")
    print(arr.transpose(0, 2, 1))

    print("")
    arr = np.arange(120).reshape(2, 3, 4, 5)
    print(arr)
    print("")
    arr1 = arr.transpose(0, 3, 2, 1) # or arr1 = arr.swapaxes(1, 3)
    print(arr1)

def element_wise_function():
    arr = np.arange(4)
    print(arr)
    # unary function
    print(np.sqrt(arr))   # same as arr ** 0.5
    print(np.square(arr)) # same as arr ** 2
    print(np.sin(arr))

    # binary function
    print(arr + arr)
    print(arr * -1.0)
    print(arr * arr)

def statistics_operation():
    arr = np.random.randn(24).reshape(2, 3, 4)
    print(arr)
    print("arr.mean(): %f" % arr.mean()) # same as np.mean(arr)
    print("arr.mean(axis = 2):")
    print(arr.mean(axis = 2)) # same as mp.mean(arr, axis = 2)
    print("np.mean(arr, axis = 1)") # same as arr.mean(axis = 1)
    print(np.mean(arr, axis = 1))

    print("np.argmax(arr, axis = 0)")
    print(np.argmax(arr, axis = 0))

def sort():
    arr = np.random.randn(5, 3)
    print(arr)
    # inplace sort, default axis is -1
    arr.sort()
    print("arr.sort(), sort along axis -1, now arr is:")
    print(arr)

    print("")
    # np.sort() returns another copy of the data
    arr1 = np.sort(arr, axis = 0)
    print("np.sort(arr, axis = 0)")
    print(arr1)

def random_number_generation():
    samples = np.random.normal(0.0, 2.0, size = (10, 10)) # generate gaussian with mean = 0.0 and stdev = 2.0
    print("generate gaussian with mean = 0.0 and stdev = 2.0, np.random.normal(0.0, 2.0, size = (3, 3):")
    print(samples)
    print("samples.mean() %f, samples.std() %f" % (samples.mean(), samples.std()))

    print("")
    samples = np.random.randint(1, 10, size = 5) # uniform distributed integer between 1 and 10
    print("uniform distributed integer between 1 and 10, np.random.randint(1, 10)")
    print(samples)

    print("")
    samples = np.random.uniform(1.0, 2.0, size = (4, 5)) # uniform distributed floating number between 1.0 and 2.0
    print("uniform distributed floating number between 1.0 and 2.0, np.random.uniform(1.0, 2.0, size = (4, 5)")
    print(samples)

if __name__ == "__main__":
    array_creation()
    print("=============================================================")
    array_data_types()
    print("=============================================================")
    array_index_and_sclice()
    print("=============================================================")
    boolean_index()
    print("=============================================================")
    fancy_index()
    print("=============================================================")
    axis_transpose()
    print("=============================================================")
    element_wise_function()
    print("=============================================================")
    statistics_operation()
    print("=============================================================")
    sort()
    print("=============================================================")
    random_number_generation()
