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

class SurpriseBasedModel:
    def __init__(self, model, name):
        self.name = name
        self.model = model

    def fit(self, trainset, testset):
        # train and test algorithm.
        algo = self.model()

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

    print("Fit each model")
    progress = tqdm(models)
    for m in progress:
        progress.set_description(m.name)
        m.fit(trainset, testset)

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

