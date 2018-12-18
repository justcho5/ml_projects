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

        for model in model_to_param:
            print("Start: {}".format(model))
            start_time = time.time()

            single_model_parameter = {
                model : model_to_param[model]
            }
            result = m.cross_validate( pool=pool,
                                       model_to_param=single_model_parameter,
                                       output_file_name=model,
                                       data_file=FILE_NAME)

            diff =  (time.time() - start_time)
            print("Time taken: {} {}s".format(model, diff))

    print("Script ended")

if __name__ == "__main__":
    with_default_param()
