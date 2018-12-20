import pandas as pd
import numpy as np
import pickle

from itertools import groupby
import scipy
import scipy.io
import scipy.sparse as sp
import surprise as s
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SlopeOne
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import KNNWithZScore
from surprise import CoClustering

from surprise import BaselineOnly
from surprise.model_selection import KFold
from sklearn.metrics import mean_squared_error
from surprise.model_selection import GridSearchCV

from sklearn import decomposition
from scipy.optimize import minimize

import scipy.sparse as sp

import dataset as d
import os
from tqdm import tqdm


def statistics(data):
    row = set([int(line[0]) for line in data])
    col = set([int(line[1]) for line in data])
    return min(row), max(row), min(col), max(col)


def to_matrix(testset, dimension):
    ratings = sp.lil_matrix((dimension))
    for row, col, rating in testset:
        ratings[int(row) - 1, int(col) - 1] = int(rating)

    return ratings


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0

    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2

    return np.sqrt(1.0 * mse / len(nz))


##
def init_MF2(train, num_features):
    model = decomposition.NMF(n_components=num_features)
    item_features = model.fit_transform(train)
    user_features = model.components_
    return user_features, item_features.T


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    return user_features, item_features


class MatrixFactor:
    def __init__(self):
        self.name = "MatrixFactor"

    def fit(self, trainset, testset, param):
        if testset is None:
            min_row, max_row, min_col, max_col = statistics(list(trainset.all_ratings()))
            dim = (max_row, max_col)
        else:
            min_row, max_row, min_col, max_col = statistics(testset)
            min_row2, max_row2, min_col2, max_col2 = statistics(list(trainset.all_ratings()))
            dim = (max(max_row, max_row2), max(max_col, max_col2))
            test = to_matrix(testset, dim)

        print("Dimension:", dim)
        train = to_matrix(list(trainset.all_ratings()), dim)

        """matrix factorization by SGD."""
        # define parameters

        # These are default parameters, expect # of num of epoch to 10

        gamma = 0.01
        num_features = 20  # K in the lecture notes
        lambda_user = 0.1
        lambda_item = 0.7
        num_epochs = 1  # number of full passes through the train set
        errors = [0]

        # init matrix
        user_features, item_features = init_MF(train, num_features)

        # find the non-zero ratings indices
        nz_row, nz_col = train.nonzero()
        nz_train = list(zip(nz_row, nz_col))

        print("learn the matrix factorization using SGD...")
        for it in range(num_epochs):
            # shuffle the training rating indices
            np.random.shuffle(nz_train)

            # decrease step size
            gamma /= 1.2

            for d, n in nz_train:
                # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
                item_info = item_features[:, d]
                user_info = user_features[:, n]
                err = train[d, n] - user_info.T.dot(item_info)

                # calculate the gradient and update
                item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
                user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

            rmse = compute_error(train, user_features, item_features, nz_train)
            print("iter: {}, RMSE on training set: {}.".format(it, rmse))

            errors.append(rmse)

        # evaluate the test error
        self.user_features = np.nan_to_num(user_features)
        self.item_features = np.nan_to_num(item_features)

        if testset is not None:
            nz_row, nz_col = test.nonzero()
            nz_test = list(zip(nz_row, nz_col))
            print("Get test RMSE")
            self.rmse = compute_error(test, user_features, item_features, nz_test)
            print("RMSE on test data: {}.".format(self.rmse))

    def predict(self, user, movie):
        item_info = self.item_features[:, int(user) - 1]
        user_info = self.user_features[:, int(movie) - 1]

        result = user_info.T.dot(item_info)
        return result


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]

        # update column row of user features
        V = M @ train[items, user]
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, V)
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        V = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, V)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features


class ALS:
    def __init__(self):
        self.name = "ALS"

    def fit(self, trainset, testset, param):
        print("Building the matrix")

        if testset is None:
            min_row, max_row, min_col, max_col = statistics(list(trainset.all_ratings()))
            dim = (max_row, max_col)
        else:
            min_row, max_row, min_col, max_col = statistics(testset)
            min_row2, max_row2, min_col2, max_col2 = statistics(list(trainset.all_ratings()))
            dim = (max(max_row, max_row2), max(max_col, max_col2))
            test = to_matrix(testset, dim)

        train = to_matrix(list(trainset.all_ratings()), dim)

        """Alternating Least Squares (ALS) algorithm."""
        num_features = 20  # K in the lecture notes
        lambda_user = 0.1
        lambda_item = 0.7
        stop_criterion = 1e-4
        change = 1
        error_list = [0, 0]

        # init ALS
        user_features, item_features = init_MF(train, num_features)

        # get the number of non-zero ratings for each user and item
        nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)

        # group the indices by row or column index
        nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

        # run ALS
        print("start the ALS algorithm...")
        while change > stop_criterion:
            # update user feature & item feature
            user_features = update_user_feature(
                train, item_features, lambda_user,
                nnz_items_per_user, nz_user_itemindices)
            item_features = update_item_feature(
                train, user_features, lambda_item,
                nnz_users_per_item, nz_item_userindices)

            error = compute_error(train, user_features, item_features, nz_train)
            print("RMSE on training set: {}.".format(error))
            error_list.append(error)
            change = np.fabs(error_list[-1] - error_list[-2])

        self.user_features = user_features
        self.item_features = item_features

        if testset is not None:
            # evaluate the test error
            nnz_row, nnz_col = test.nonzero()
            nnz_test = list(zip(nnz_row, nnz_col))
            self.rmse = compute_error(test, user_features, item_features, nnz_test)
            print("test RMSE after running ALS: {v}.".format(v=rmse))

    def predict(self, user, movie):
        item_info = self.item_features[:, int(user) - 1]
        user_info = self.user_features[:, int(movie) - 1]

        return user_info.T.dot(item_info)
