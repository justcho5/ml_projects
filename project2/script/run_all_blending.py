# -*- coding: utf-8 -*-
import surprise
import numpy as np
import pandas as pd
import pickle

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import SVD
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

        output_file_name = "model_selection_blending"

        start_time = time.time()

        all = m.cross_validate(pool=pool,
                               splits=12,
                               model_to_param=model_to_param,
                               output_file_name=output_file_name,
                               data_file=FILE_NAME,
                               with_blending=True)

        best = min(all, key=lambda each: each[1].fun)
        models = best[0]
        weights = best[1].x

        s.create_submission(best, models, output_file_name, pool, weights)

        diff = (time.time() - start_time)
        print("Time taken: {}s".format(diff))

        print("Script ended")



if __name__ == "__main__":
    with_default_param()
