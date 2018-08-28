#!/usr/bin/env python3
# coding=utf8

import math
import numpy as np
import random
from sklearn import mixture
import matplotlib.pyplot as plt


def generateGMMSamples(n_samples, means_, covs_, priors_):
    n_gaussians, feature_dim = means_.shape
    samples = np.empty((n_samples, feature_dim), dtype=means.dtype)

    ## randomly choose n_samples number ranged in [0, n_gaussians) with specified probabilities priors_ for each number
    which_gaussians = np.random.choice(range(n_gaussians), n_samples, p=priors_)
    for i, gaussian_index in enumerate(which_gaussians):
        ## sample from a specified gaussian
        samples[i] = np.random.multivariate_normal(means_[gaussian_index], covs_[gaussian_index])

    return samples


## refers to http://www.vlfeat.org/overview/gmm.html, the section "Initializing a GMM model before running EM"
## The EM algorithm is a local optimization method, and hence particularly sensitive to the
## initialization of the model. The simplest way to initiate the GMM is to pick numClusters
## data points at random as mode means, initialize the individual covariances as the covariance
## of the data, and assign equal prior probabilities to the modes.
def gmmInitialize(data, num_gaussians):
    num_samples, feature_dim = data.shape

    ## randomly select num_gaussians unique number ranged in [0, num_samles)
    rand_indices = random.sample(range(num_samples), num_gaussians)

    ## initialize the mean vectors with the ramdomly selected data sample
    mean_vectors = data[rand_indices].copy()

    ## coveriance matrix of the training data
    data_cov = np.cov(np.transpose(data))

    cov_matrices = np.empty((num_gaussians, feature_dim, feature_dim), dtype=data.dtype)
    ## use data coveriance matrix to initialize all coverance matrices
    cov_matrices[:] = data_cov

    ## initially each gaussian has equal probability
    laten_rv_priors = np.ones(num_gaussians, dtype=data.dtype) / num_gaussians

    return mean_vectors, cov_matrices, laten_rv_priors


## Implementation of GMM training,
## according to Andrew NG's CS229 lecture notes "Mixtures of Gaussians and The EM algorithm"
def gmmTrain(data, num_gaussians, max_iterations=10):
    num_samples, feature_dim = data.shape

    ## initialize parameters
    mean_vectors, cov_matrices, latent_rv_priors = gmmInitialize(data, num_gaussians)

    w = np.zeros((num_samples, num_gaussians), dtype=data.dtype)
    k = math.sqrt(math.pow(2.0 * math.pi, feature_dim))
    for iter_index in range(max_iterations):
        ## shuffle the training data
        np.random.shuffle(data)

        ##########################################################################################
        ## ============== E-step ==============
        for i, x in enumerate(data):
            for j in range(0, num_gaussians):
                ## evaluate the exponent part of the j-th gaussian pdf
                v = np.exp(-0.5 * np.dot(np.dot(x - mean_vectors[j], np.linalg.inv(cov_matrices[j])),
                                         np.transpose(x - mean_vectors[j])))
                ## evaluate the normalization part of the j-th gaussian pdf
                f = 1.0 / (k * math.sqrt(np.linalg.det(cov_matrices[j])))

                w[i, j] = f * v * latent_rv_priors[j]

        # normalize w along the row
        w_row_sum = np.sum(w, axis=1)
        w /= w_row_sum[:, np.newaxis]
        ##########################################################################################

        ##########################################################################################
        ## ============== M-step ==============
        ## average over data samples
        latent_rv_priors = np.mean(w, axis=0)

        for j in range(num_gaussians):
            mean_vectors[j] = np.sum(w[:, j:j + 1] * data, axis=0)
        w_col_sum = np.sum(w, axis=0)
        mean_vectors /= w_col_sum[:, np.newaxis]

        for j in range(num_gaussians):
            data_delta = data - mean_vectors[j]
            cov_matrices[j] = np.dot(np.transpose(data_delta), w[:, j:j + 1] * data_delta)
            cov_matrices[j] /= w_col_sum[j]
            ##########################################################################################

        # print("")
        # print("============= iteration %d =============" % (iter_index,))
        # print("latent_rv_priors:")
        # print(latent_rv_priors)
        # print()
        # print("mean_vectors:")
        # print(mean_vectors)
        # print()
        # print("cov_matrices:")
        # print(cov_matrices)

    return latent_rv_priors, mean_vectors, cov_matrices


if __name__ == "__main__":
    data_type = np.float

    means = np.array([
        [-10, -10],
        [0, 0],
        [10, 10]
    ], dtype=data_type)

    covs = np.array([
        [[1.0, 0.0],
         [0.0, 1.0]],
        [[2.0, 0.0],
         [0.0, 2.0]],
        [[3.0, 0.0],
         [0.0, 3.0]]
    ], dtype=data_type)

    priors = np.array([0.4, 0.1, 0.5], dtype=data_type)

    X = generateGMMSamples(8000, means, covs, priors)
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.show()

    ################################################################################
    ## use my demo implementation of GMM training to fit the data
    priors_predicted, means_predicted, cov_matrices_predicted = gmmTrain(X, 3, 200)

    print("**************************************************************")
    print("=================== my prediction ===================")
    print("priors:")
    print(priors_predicted)
    print()

    print("means:")
    print(means_predicted)
    print()

    print("cov_matrices:")
    print(cov_matrices_predicted)
    print("**************************************************************")
    ################################################################################

    print()

    ##################################################################################################
    ## use sklearn.mixture.GaussianMixture to fit the data
    clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
    clf.fit(X)
    print("**************************************************************************************")
    print("=================== sklearn.mixture.GaussianMixture prediction ===================")
    print("priors:")
    print(clf.weights_)
    print()
    print("means:")
    print(clf.means_)
    print()
    print("cov_matrices:")
    print(clf.covariances_)
    print("**************************************************************************************")
    ##################################################################################################
