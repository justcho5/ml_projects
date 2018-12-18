# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle

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


import dataset as d

import os

from tqdm import tqdm


# from tqdm import tqdm_notebook
# tqdm = tqdm_notebook

# class Dataset:
# def __init(self, full_data):
# self.full_data = full_data

class SurpriseBaselineOnly:
    def __init__(self):
        self.name = 'BaselineOnly'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = BaselineOnly()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseCoClustering:
    def __init__(self):
        self.name = 'CoClustering'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = CoClustering()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseKNNBaseline:
    def __init__(self):
        self.name = 'KNNBaseline'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = KNNBaseline()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseKNNWithZScore:
    def __init__(self):
        self.name = 'SurpriseKNNWithZScore'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = SurpriseKNNWithZScore()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseKNNWithMeans:
    def __init__(self):
        self.name = 'KNNWithMeans'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = KNNWithMeans()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)

        self.algo = algo


class SurpriseKNNBasic:
    def __init__(self):
        self.name = 'KNNBasic'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = KNNBasic()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseNMF:
    def __init__(self):
        self.name = 'NMF'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = NMF()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseSvdPPModel:
    def __init__(self):
        self.name = 'Surprise SVDpp'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = SVDpp()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseSvdModel:
    def __init__(self):
        self.name = 'Surprise SVD'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = SVD()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo


class SurpriseSlopeOneModel:
    def __init__(self):
        self.name = 'Surprise SlopeOne'

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = SlopeOne()

        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

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
            p = each_model.algo.predict(each_data[0], each_data[1]).est
            predictions.append(p)
        result.append(predictions)
    return result


def call_algo(i):
    trainset, testset, model_name = i

    models = []

    if "SurpriseSlopeOneModel" in model_name:
        models.append(SurpriseSlopeOneModel())

    if "SurpriseSvdModel" in model_name:
        models.append(SurpriseSvdModel())

    if "SurpriseSvdPPModel" in model_name:
        models.append(SurpriseSvdPPModel())

    if "SurpriseNMF" in model_name:
        models.append(SurpriseNMF())

    if "SurpriseKNNBasic" in model_name:
        models.append(SurpriseKNNBasic())

    if "SurpriseKNNWithMeans" in model_name:
        models.append(SurpriseKNNWithMeans())

    if "SurpriseKNNBaseline" in model_name:
        models.append(SurpriseKNNBaseline())

    if "SurpriseCoClustering" in model_name:
        models.append(SurpriseCoClustering())

    if "SurpriseBaselineOnly" in model_name:
        models.append(SurpriseBaselineOnly())

    if "SurpriseBaselineOnly" in model_name:
        models.append(SurpriseBaselineOnly())

    print("Fit models")
    t = tqdm(models)
    for m in t:
        t.set_description(m.name)
        m.fit(trainset, testset)


    # do blening
    print("Do blending")
    model_predictions = get_predictions(models, testset)
    real = list(map(lambda x: x[2], testset))

    w0 = [1 / len(models)] * len(models)
    result = minimize(fun=calcualte_mean_square_error, x0=w0,
                      args=(model_predictions, real),
                      options={'maxiter': 1000, 'disp': True})

    print("Best blended rmse: ", result.fun)
    return (models, result)


def cross_validate(pool, whole_data, is_parallel=True):
    kf = KFold(n_splits=8)

    results = []
    splits = list(kf.split(whole_data))

    print("running CV")
    ## run the code sequentially or parallely
    if is_parallel:
        x = list(map(lambda x: (whole_data[x[0]], whole_data[x[1]]), splits))
        for result in tqdm(pool.imap(call_algo, x), total=len(x), desc="CV"):
            results.append(result)
    else:
        for train, test in tqdm(splits):
            trainset = whole_data[train]
            testset = whole_data[test]
            train_test = (trainset, testset)
            results.append(call_algo(train_test))

    for m in range(len(results[0])):
        for r in results:
            print(r[m].name, r[m].rmse)

    return results


def cross_validates_one_by_one(pool, model_name,
                               path='../data/data_surprise.csv',
                               splits = 12):

    print("Running with models '{}' and split {}".format(model_name, splits))
    data = d.to_surprise_read(path)
    kf = KFold(n_splits=splits)

    results = []
    print("Split data")
    splits = list(kf.split(data))

    print("running CV")
    ## run the code sequentially or parallely
    x = list(map(lambda x: (x[0], x[1], model_name), splits))
    for result in tqdm(pool.imap(call_algo, x), total=len(x), desc="CV"):
        results.append(result)

    #for m in range(len(results[0])):
        #for r in results:
            #print(r)
            #print(r[m].name, r[m].rmse)

    if not os.path.exists("result"):
        os.makedirs("result")
    pickle.dump(results, open("result/out.result", "wb"))
    return results


def grid_search():
    data = d.to_surprise_read('../data/data_surprise.csv')

    print("Start grid search for KNNBaseline")
    crazy_param = {
        'k': np.arange(1, 100, 10),
    }

    gs = GridSearchCV(KNNBasic,
                      crazy_param,
                      return_train_measures=True,
                      measures=['rmse'], cv=12, n_jobs=16, joblib_verbose=True)
    gs.fit(data)
    pickle.dump(gs, open("output/GridSearch_KNNBaseline.result", "wb"))
    return gs
