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

def with_default_param():
    print("Start script");

    with Pool(12) as pool:
        model_to_param = {
            #BaselineOnly": {},
            #SVD": { 'n_factors': 20 },
            #SlopeOne": {},
            #KNNBaseline":   {
            #   'k': 100,
            #   'sim_options': { 'name': 'pearson_baseline', 'user_based': 'True' }
            #,
            #GlobalMean": {},
            #UserMean": {},
            #"MovieMean": {},
            "MatrixFactor": {},
             #"KNNWithMeans": {},
             #"KNNWithZScore": {},
             #"KNNBasic": {},
            #"SVDpp": {},
            #"NMF": {},
            #"CoClustering": {},
        }
        output_file_name = "blending_with_all"

        start_time = time.time()

        all = m.cross_validate(pool=pool,
                               splits=4,
                               model_to_param=model_to_param,
                               output_file_name=output_file_name,
                               data_file=FILE_NAME,
                               with_blending=True)

        print("Find best model")
        best = min(all, key=lambda each: each[1].fun)

        models = best[0]
        weights = best[1].x
        print("Weights: ", list(zip(map(lambda x: x.name, models), weights)))

        print("Best rmse: ", best[1].fun)

        print("Read submission file")
        df_submission = pd.read_csv(SAMPLE_SUBMISSION)
        df_submission = split_user_movie(df_submission)

        print("Do predictions")
        predictions = []
        items_to_predict = list(df_submission.iterrows())

        items = np.array_split(items_to_predict, 12)
        items = map(lambda x: (x, models, predictions, weights), items)

        print("Start jobs")
        p = tqdm(pool.imap(predict, items), total=12)
        new_predictions = [item for sublist in p for item in sublist]

        print("Create File")
        s.write_predictions_to_file(new_predictions, output_file_name + "_prediction.csv")

        diff = (time.time() - start_time)
        print("Time taken: {}s".format(diff))

        print("Script ended")


def predict(input):
    items_to_predict, models, predictions, weights = input
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
    with_default_param()
