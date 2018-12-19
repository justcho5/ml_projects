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
            "BaselineOnly": {},
            "KNNBasic":  {},
            "SVD": {},  #{ 'n_factors': 20 }
            "SlopeOne": {},

            # {
            #     'k': 100,
            #     'sim_options': {
            #         'name': 'pearson_baseline',
            #         'user_based': 'True'
            #     }
            # }
            # "KNNWithMeans": {},
            # "KNNWithZScore": {},
            # "KNNBaseline": {},
            # "SVDpp": {},
            # "NMF": {},
            # "CoClustering": {},
        }
        output_file_name = "all_blending"

        start_time = time.time()

        all = m.cross_validate(pool=pool,
                               model_to_param=model_to_param,
                               output_file_name=output_file_name,
                               data_file=FILE_NAME,
                               with_blending=True)

        print("Find best model")
        best = min(all, key=lambda each: each[1].fun)

        models = best[0]
        weights = best[1].x
        print("Best rmse: ", best[1].fun)

        print("Read submission file")
        df_submission = pd.read_csv(SAMPLE_SUBMISSION)
        df_submission = split_user_movie(df_submission)

        print("Do predictions")
        predictions = []
        for each in tqdm(df_submission.iterrows(), total=len(df_submission)):
            user = each[1].user
            movie = each[1].movie

            mix_prediction = 0
            for i, w in enumerate(models):
                pred = w.algo.predict(user, movie).est
                mix_prediction += weights[i] * pred
            predictions.append(mix_prediction)
        clipped_predictions = np.clip(predictions, 1, 5)

        print("Do predictions")
        new_predictions = []
        for i, each in enumerate(tqdm(df_submission.iterrows(), total=len(df_submission))):
            one = (each[1].user, each[1].movie, clipped_predictions[i])
            new_predictions.append(one)

        print("Do predictions")
        s.write_predictions_to_file(new_predictions, output_file_name + "_prediction.csv")

        diff = (time.time() - start_time)
        print("Time taken: {}s".format(diff))

        print("Script ended")

if __name__ == "__main__":
    with_default_param()
