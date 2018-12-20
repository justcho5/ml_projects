# -*- coding: utf-8 -*-
'''
This file was used during gridsearch to find the best parameter.
'''


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


def grid_search():
   data = d.to_surprise_read('../data/data_surprise.csv')

   print("Start grid search for KNNBaseline")
   crazy_param = {
      'k': np.arange(1, 100, 10),
   }

   gs = GridSearchCV(KNNBasic,
                     crazy_param,
                     return_train_measures=True,
                     measures=['rmse'], cv=12, n_jobs=16, joblib_verbose=True)
   gs.fit(data)
   pickle.dump(gs, open("output/GridSearch_KNNBaseline.result", "wb"))
   return ge


if __name__ == "__main__":
   grid_search()
