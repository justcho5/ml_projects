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
import time
import sys

FILE_NAME = '../data/data_surprise.csv'


def with_default_param():
    print("Start script");

    with Pool(12) as pool:
        model_to_param = {
            "BaselineOnly": {

            },
            "KNNBasic": {
                'k': 100,
                'sim_options': {
                    'name': 'pearson_baseline',
                    'user_based': 'True'
                }
            },
            # "KNNWithMeans": {},
            # "KNNWithZScore": {},
            # "KNNBaseline": {},
            "SVD": {
                'n_factors': 20
            },
            # "SVDpp": {},
            # "NMF": {},
            "SlopeOne": {

            },
            # "CoClustering": {},
        }

        start_time = time.time()

        all = m.cross_validate(pool=pool,
                               model_to_param=model_to_param,
                               output_file_name="all_blending",
                               data_file=FILE_NAME)

        diff = (time.time() - start_time)
        print("Time taken: {}s".format(diff))

        print("Script ended")

if __name__ == "__main__":
    with_default_param()
