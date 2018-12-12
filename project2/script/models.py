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
from surprise import CoClustering
from surprise import BaselineOnly
from surprise.model_selection import KFold
from sklearn.metrics import mean_squared_error

import os

from tqdm import tqdm

#from tqdm import tqdm_notebook
#tqdm = tqdm_notebook

#class Dataset:
    #def __init(self, full_data):
        #self.full_data = full_data

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

    for m in models:
        m.fit(trainset, testset)

    return models

def cross_validate(pool, whole_data, is_parallel=True):
    kf = KFold(n_splits = 8)

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

def cross_validates_one_by_one(pool, whole_data, model_name):
    file_path = '../data/data_surprise.csv'
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    kf = KFold(n_splits = 12)

    results = []
    splits = list(kf.split(data))

    print("running CV")
    ## run the code sequentially or parallely
    x = list(map(lambda x : (x[0], x[1], model_name), splits))
    for result in tqdm(pool.imap(call_algo, x), total=len(x), desc="CV"):
        results.append(result)

    for m in range(len(results[0])):
        for r in results:
            print(r[m].name, r[m].rmse)

    if not os.path.exists("result"):
        os.makedirs("result")

    pickle.dump(results, open("result/" + model_name + ".result", "wb"))
    return results



