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
            "KNNBasic": {},
            "KNNWithMeans": {},
            "KNNWithZScore": {},
            "KNNBaseline": {},
            "SVD": {},
            "SVDpp": {},
            "NMF": {},
            "SlopeOne": {},
            "CoClustering": {},
        }

        start_time = time.time()

        lower_result = m.cross_validate(pool=pool,
                                        model_to_param=model_to_param,
                                        output_file_name="lower",
                                        data_file=FILE_LOWER)

        upper_result = m.cross_validate(pool=pool,
                                        model_to_param=model_to_param,
                                        output_file_name="upper",
                                        data_file=FILE_LOWER)

        diff = (time.time() - start_time)
        print("Time taken: {}s".format(diff))

def prediction_with_two(left_result, right_result):




erint("Script ended")

if __name__ == "__main__":
    with_default_param()
