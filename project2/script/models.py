# -*- coding: utf-8 -*-
'''
This is the  main file, which contains all models used for this project.
'''

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

from scipy.optimize import minimize

import scipy.sparse as sp

import dataset as d
import os
from tqdm import tqdm

def statistics(data):
    row = set([int(line[0]) for line in data])
    col = set([int(line[1]) for line in data])
    return min(row), max(row), min(col), max(col)

def to_matrix(testset):
    '''
    form hw
    '''
    min_row, max_row, min_col, max_col = statistics(testset)

    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in testset:
        ratings[int(row) - 1, int(col) - 1] = int(rating)

    return ratings

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

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    pred_matrix = user_features.T @ item_features
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2

    return np.sqrt(1.0 * mse / len(nz))

class MatrixFactor:
    def __init__(self):
        self.name = "MatrixFactor"

    def fit(self, trainset, testset, param):
        print("Building the matrix")
        test = to_matrix(testset)
        train = to_matrix(list(trainset.all_ratings()))

        """matrix factorization by SGD."""
        # define parameters
        gamma = 0.01
        num_features = 20  # K in the lecture notes
        lambda_user = 0.1
        lambda_item = 0.7
        num_epochs = 20  # number of full passes through the train set
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
        self.user_features = user_features
        self.item_features = item_features

        nz_row, nz_col = test.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        self.rmse = compute_error(test, user_features, item_features, nz_test)
        print("RMSE on test data: {}.".format(rmse))

    def predict(self, user, movie):
        item_info = item_features[:, int(user)]
        user_info = user_features[:, int(movie)]

        return user_info.T.dot(item_info)

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

        train = to_matrix(list(trainset.all_ratings()))
        test = to_matrix(testset)

        """Alternating Least Squares (ALS) algorithm."""
        # define parameters
        num_features = 20  # K in the lecture notes
        lambda_user = 0.1
        lambda_item = 0.7
        stop_criterion = 1e-2
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

        # evaluate the test error
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        rmse = compute_error(test, user_features, item_features, nnz_test)
        print("test RMSE after running ALS: {v}.".format(v=rmse))

    def predict(self, user, movie):
        item_info = item_features[:, int(user)]
        user_info = user_features[:, int(movie)]

        return user_info.T.dot(item_info)

class GlobalMean:
    ''''
    Simple Model: always return the global mean
    '''
    def __init__(self):
        self.name = "GlobalMean"

    def fit(self, trainset, testset, param):

        # get all ratings as list
        only_ratings_train = list(map(lambda x: x[2], trainset.all_ratings()))

        # calculate mean of all ratings
        global_mean = np.mean(only_ratings_train)

        # get all test ratings
        only_ratings_test = list(map(lambda x: x[2], testset))

        # our predictions is just the global mean
        predictions = np.repeat(global_mean, len(testset))

        self.rmse = np.sqrt(mean_squared_error(only_ratings_test, predictions))
        self.global_mean = global_mean

    def predict(self, user, movie):
        return self.global_mean

class UserMean:
    ''''
    Simple Model:
    Find the users mean over all his ratings and use that as the prediction
    '''
    def __init__(self):
        self.name = "UserMean"

    def fit(self, trainset, testset, param):

        df_train = pd.DataFrame.from_records(trainset.all_ratings(), columns=['user', 'movie', 'rating'])

        # this is strange, surprise seems to index user by 0
        df_train.user = df_train.user + 1

        # get mean per user
        self.df_user_to_rating = df_train.groupby('user')['rating'].mean()

        # get mean rating per user
        predictions = []
        true_rating = []
        for each in testset:
            user = each[0]
            rating = each[2]
            predictions.append(self.df_user_to_rating.loc[int(user)])
            true_rating.append(rating)

        self.rmse = np.sqrt(mean_squared_error(true_rating, predictions))

    def predict(self, user, movie):
        return self.df_user_to_rating.loc[int(user)]

class MovieMean:
    ''''
    Find the movie mean over all movie-ratings and use that as the prediction
    '''
    def __init__(self):
        self.name = "MovieMean"

    def fit(self, trainset, testset, param):

        df_train = pd.DataFrame.from_records(trainset.all_ratings(), columns=['user', 'movie', 'rating'])
        df_train.movie = df_train.movie + 1

        # get mean per movie
        self.df_movie_to_rating = df_train.groupby('movie')['rating'].mean()

        predictions = []
        true_rating = []
        for each in testset:
            movie = each[1]
            rating = each[2]
            predictions.append(self.df_movie_to_rating.loc[int(movie)])
            true_rating.append(rating)

        self.rmse = np.sqrt(mean_squared_error(true_rating, predictions))

    def predict(self, user, movie):
        return self.df_movie_to_rating.loc[int(movie)]

class SurpriseBasedModel:
    '''
    This is a generic class in order to use the models provided
    by the surprise library.
    '''
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def fit(self, trainset, testset, param):

        # These additonal parameter given to the model
        if param:
            # train and test algorithm.
            algo = self.model(**param)
        else:
            algo = self.model()

        # fit the model and predict for test result
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

    def predict(self, user, movie):
        return self.algo.predict(user, movie).est

def calcualte_mean_square_error(weights, model_predictions, real):
    preds = []
    for i, pred in enumerate(model_predictions):
        mix_prediction = 0
        for i, w in enumerate(weights):
            mix_prediction += weights[i] * pred[i]
        preds.append(mix_prediction)
    preds = np.array(preds)
    preds = preds.clip(1, 5)

    mse = mean_squared_error(preds, real)
    return np.sqrt(mse)


def get_predictions(models, data_to_predict):
    result = []
    for each_data in tqdm(data_to_predict):
        predictions = []
        for each_model in models:
            p = each_model.predict(each_data[0], each_data[1])
            predictions.append(p)
        result.append(predictions)
    return result


def blending_result(models, testset):
    # do blening
    if len(models) > 1:

        print("Do blending")
        model_predictions = get_predictions(models, testset)
        real = list(map(lambda x: x[2], testset))

        w0 = [1 / len(models)] * len(models)
        result = minimize(fun=calcualte_mean_square_error,
                          x0=w0,
                          args=(model_predictions, real), options={'disp': True})
        print("Best blended rmse: ", result.fun)
        return result
    else:
        return None


def call_algo(i):
    trainset, testset, model_name, with_blending = i

    models = []

    if "BaselineOnly" in model_name:
        models.append(SurpriseBasedModel(BaselineOnly, "BaselineOnly"))

    if "KNNBasic" in model_name:
        models.append(SurpriseBasedModel(KNNBasic, "KNNBasic"))

    if "KNNWithMeans" in model_name:
        models.append(SurpriseBasedModel(KNNWithMeans, "KNNWithMeans"))

    if "KNNWithZScore" in model_name:
        models.append(SurpriseBasedModel(KNNWithZScore, "KNNWithZScore"))

    if "KNNBaseline" in model_name:
        models.append(SurpriseBasedModel(KNNBaseline, "KNNBaseline"))

    if "SVD" in model_name:
        models.append(SurpriseBasedModel(SVD, "SVD"))

    if "SVDpp" in model_name:
        models.append(SurpriseBasedModel(SVDpp, "SVDpp"))

    if "NMF" in model_name:
        models.append(SurpriseBasedModel(NMF, "NMF"))

    if "SlopeOne" in model_name:
        models.append(SurpriseBasedModel(SlopeOne, "SlopeOne"))

    if "CoClustering" in model_name:
        models.append(SurpriseBasedModel(CoClustering, "CoClustering"))

    if "GlobalMean" in model_name:
        models.append(GlobalMean())

    if "UserMean" in model_name:
        models.append(UserMean())

    if "MovieMean" in model_name:
        models.append(MovieMean())

    if "MatrixFactor" in model_name:
        models.append(MatrixFactor())

    if "ALS" in model_name:
        models.append(ALS())

    print("Fit each model")
    progress = tqdm(models)
    for m in progress:
        progress.set_description(m.name)
        m.fit(trainset, testset, model_name[m.name])

    if with_blending:
        blending =  blending_result(models, testset)
    else:
        blending = None

    return (models, blending)


def cross_validate(pool,
                   model_to_param,
                   output_file_name,
                   data_file,
                   splits = 12,
                   with_blending=False):

    models = list(model_to_param.keys())
    print("Running with models '{}' and split {}".format(models, splits))

    data = d.to_surprise_read(data_file)
    kf = KFold(n_splits = splits)

    results = []
    print("Split data")
    splits = list(kf.split(data))

    print("running CV")

    ## run the code sequentially or parallely
    argument_list = list(map(lambda x: (x[0], x[1], model_to_param, with_blending), splits))

    for result in tqdm(pool.imap(call_algo, argument_list), total=len(argument_list), desc="CV"):
        results.append(result)

    if not os.path.exists("result"):
        os.makedirs("result")

    file_to_write_to = "result/{}.result".format(output_file_name)
    print("Write result to file", file_to_write_to)
    pickle.dump(results, open(file_to_write_to, "wb"))
    return results

