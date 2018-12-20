# -*- coding: utf-8 -*-
import surprise
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


def split_user_movie(pandas_data_frame):
    user_movie = pandas_data_frame.Id.str.extract(r'r(?P<user>\d+)_c(?P<movie>\d+)')
    pandas_data_frame['user'] = user_movie.user
    pandas_data_frame['movie'] = user_movie.movie
    return pandas_data_frame[['user', 'movie', 'Prediction']]

def main():
    print("Start script");

    model_to_param = {
        "BaselineOnly": {},
        "SVD": {'n_factors': 20},
        "SlopeOne": {},
        "KNNBaseline": {
           #'k': 150,
           'sim_options': {
               'name': 'pearson_baseline',
               'user_based': 'True'
           }
        },
        "GlobalMean": {},
        "UserMean": {},
        "MovieMean": {},
        "MatrixFactor": {},
        "ALS": {},
    }

    model_to_weight = {
        "BaselineOnly": -0.25334606,
        "SVD": 0.01819908,
        "SlopeOne": -0.04232111,
        "KNNBaseline": 0.404621,
        "GlobalMean": 0.04437225,
        "UserMean":  0.07933614,
        "MovieMean": -0.06444559,
        "MatrixFactor": 0.66204053,
        "ALS": 0.16096753
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
    df_submission = split_user_movie(df_submission)

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
