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
import sys

#FILE_NAME = '../data/data_train_small.csv'
FILE_NAME = '../data/data_train.csv'

def main():
    print("Start script");

    with Pool(12) as p:
        data = d.read_data(FILE_NAME)
        data = np.array(data)
        result = m.cross_validates_one_by_one(pool, data, is_parallel=True, sys.argv[1]):
        #result = m.cross_validate(p, data)

    print("Script ended")

    return (result, data)

def save_to_pickle(data):
    pickle.dump( data, open( "data.p", "wb" ) )

if __name__ == "__main__":
   data = main()
   save_to_pickle(data)
