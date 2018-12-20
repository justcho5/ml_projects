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

import matrix_factorization as mf


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
        self.global_mean = global_mean

        if testset is not None:

            # get all test ratings
            only_ratings_test = list(map(lambda x: x[2], testset))

            # our predictions is just the global mean
            predictions = np.repeat(global_mean, len(testset))

            self.rmse = np.sqrt(mean_squared_error(only_ratings_test, predictions))

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

        if testset is not None:
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

        if testset is not None:
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

        if testset is not None:
            predictions = algo.test(testset)

            # Compute and print Root Mean Squared Error
            self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

    def predict(self, user, movie):
        return self.algo.predict(user, movie).est


def calcualte_mean_square_error(weights, model_predictions, real):
    '''
    Calculates mean square error for several models combined using the given weights.

    :param weights: the weight of each model
    :param model_predictions: the predicitons by each model
    :param real: the actual value
    :return: RMSE between prediction and real value
    '''

    prediction = []

    # calculate the weighted prediction
    for i, pred in enumerate(model_predictions):
        mix_prediction = 0
        for i, w in enumerate(weights):
            mix_prediction += weights[i] * pred[i]
        prediction.append(mix_prediction)

    prediction = np.array(prediction)
    prediction = prediction.clip(1, 5)

    mse = mean_squared_error(prediction, real)
    return np.sqrt(mse)


def get_predictions(models, data_to_predict):
    '''
    :param models: list of models
    :param data_to_predict: list of tuples to predict
    :return: the predictions for all given model
    '''

    result = []
    for each_data in tqdm(data_to_predict):
        predictions = []
        for each_model in models:
            p = each_model.predict(each_data[0], each_data[1])
            predictions.append(p)
        result.append(predictions)
    return result


def blending_result(models, testset):
    '''
    Tries to find the best weight such that you can combine the predictions,
    such that the RMSE is minimized

    :param models: list of models
    :param testset: the data to test the blending on
    :return: the best weights or None
    '''

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
    '''
    This is a help construct such that you can runn it in parallel
    :param i: tuple of parameter
    :return: the best result
    '''
    trainset, testset, model_name, with_blending, data = i

    models = model_name_to_model(model_name)

    print("Fit each model")
    progress = tqdm(models)

    for each_model in progress:
        progress.set_description(each_model.name)
        each_model.fit(trainset, testset, model_name[each_model.name])

    if with_blending:
        blending =  blending_result(models, testset)
    else:
        blending = None

    return (models, blending)

def model_name_to_model(model_name):
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
        models.append(mf.MatrixFactor())
    if "ALS" in model_name:
        models.append(mf.ALS())
    return models


def cross_validate(pool,
                   model_to_param,
                   output_file_name,
                   data_file,
                   splits = 12,
                   with_blending=False):
    '''
    Helper function which does cross validation in prallel

    :param pool: The pool to use for parallelism
    :param model_to_param:
    dictionary key is the model to use and value is the model parameter

    :param output_file_name: if given, the output is persisted on the disk
    :param data_file:
    the input data file

    :param splits: how many splits to perform

    :param with_blending: return blending weights if this is True
    :return: all the results form each fold
    '''

    models = list(model_to_param.keys())
    print("Running with models '{}' and split {}".format(models, splits))

    data = d.to_surprise_read(data_file)
    kf = KFold(n_splits = splits)

    results = []
    print("Split data")
    splits = list(kf.split(data))

    print("running CV")

    ## run the code sequentially or in parlalely
    argument_list = list(map(lambda x: (x[0], x[1], model_to_param, with_blending, data), splits))

    for result in tqdm(pool.imap(call_algo, argument_list), total=len(argument_list), desc="CV"):
        results.append(result)

    persist_result(output_file_name, results)
    return results


def persist_result(output_file_name, results):
    if not os.path.exists("result"):
        os.makedirs("result")
    if output_file_name is not None:
        file_to_write_to = "result/{}.result".format(output_file_name)
        print("Write result to file", file_to_write_to)
        pickle.dump(results, open(file_to_write_to, "wb"))

