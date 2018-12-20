# -*- coding: utf-8 -*-
import surprise
import random
import numpy as np
import pandas as pd
import pickle

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import KNNWithMeans
from surprise import SlopeOne
from tqdm import tqdm_notebook, tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from surprise import accuracy
from surprise.model_selection import KFold
from multiprocessing import Pool

# our module imports
import dataset as d
import models as m
import submission as s
import time
import sys

FILE_NAME = '../data/data_surprise.csv'
SAMPLE_SUBMISSION = '../data/sample_submission.csv'


def main():
    print("Start script");
    start_time = time.time()
    random.seed(1000)

    model_to_param = {
        "BaselineOnly": {},
        "SVD": {},
        "SlopeOne": {},
        "KNNBaseline": {
            'k': 150,
        },
        "KNNBasic": {},
        "NMF": {},
    }

    model_to_weight = {
        "BaselineOnly": -0.7855352289256156,
        "SVD": 0.26101515307439144,
        "SlopeOne": 0.9527865189392614,
        "KNNBaseline": 0.47538566082270134,
        "KNNBasic": 0.13607754030687072,
        "NMF": -0.033297000709009685,
    }

    print("Models: ", model_to_param)
    print("Weights", model_to_weight)

    print("Read file")
    full_data = d.to_surprise_read(FILE_NAME)
    trainset = full_data.build_full_trainset()

    models = m.model_name_to_model(model_to_param)
    for each_model in models:
        print("Fit momdel", each_model.name)
        each_model.fit(trainset, None, model_to_param[each_model.name])

    weights = list(model_to_weight.values())
    output_file_name = "reproducable_submission"
    create_submission(models, output_file_name, weights)

    diff = (time.time() - start_time)
    print("Time taken: {}s".format(diff))

    print("Script ended")


def create_submission(models, output_file_name, weights):
    print("Read submission file")
    df_submission = pd.read_csv(SAMPLE_SUBMISSION)
    df_submission = d.split_user_movie(df_submission)

    print("Do predictions in parallel")
    items_to_predict = list(df_submission.iterrows())

    print("Split data")
    pool_size = 12
    items = np.array_split(items_to_predict, pool_size)
    items = map(lambda x: (x, models, weights), items)

    print("Start jobs")
    with Pool(pool_size) as pool:
        p = tqdm(pool.imap(predict, items), total=pool_size)
        new_predictions = [item for sublist in p for item in sublist]

        print("Create File")
        s.write_predictions_to_file(new_predictions, output_file_name + "_prediction.csv")

def predict(input):
    items_to_predict, models, weights = input

    predictions = []
    print("Predict for", len(items_to_predict))
    for each in tqdm(items_to_predict):
        user = each[1].user
        movie = each[1].movie

        mix_prediction = 0
        for i, w in enumerate(models):
            pred = w.predict(user, movie)
            mix_prediction += weights[i] * pred
        predictions.append(mix_prediction)
    clipped_predictions = np.clip(predictions, 1, 5)

    new_predictions = []
    for i, each in enumerate(items_to_predict):
        one = (each[1].user, each[1].movie, clipped_predictions[i])
        new_predictions.append(one)

    return new_predictions


if __name__ == "__main__":
    main()
