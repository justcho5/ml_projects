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

FILE_LOWER = '../data/lower_surprise.csv'
FILE_UPPER = '../data/upper_surprise.csv'


def with_default_param():
    print("Start script");

    with Pool(12) as pool:
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

        start_time = time.time()
        lower_result = m.cross_validate(pool=pool,
                                        model_to_param=model_to_param,
                                        output_file_name="lower",
                                        data_file=FILE_LOWER,
                                        with_blending=True)

        upper_result = m.cross_validate(pool=pool,
                                        model_to_param=model_to_param,
                                        output_file_name="upper",
                                        data_file=FILE_LOWER,
                                        with_blending=True)

        best = min(lower_result, key=lambda each: each[1].fun)
        models = best[0]
        weights = best[1].x


        print(lower_result)
        diff = (time.time() - start_time)
        print("Time taken: {}s".format(diff))


if __name__ == "__main__":
    with_default_param()
