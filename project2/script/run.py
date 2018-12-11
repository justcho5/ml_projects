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

#FILE_NAME = '../data/data_train_small.csv'
FILE_NAME = '../data/data_train.csv'

def main():
    print("Start script");

    with Pool(12) as p:
        data = d.read_data(FILE_NAME)
        data = np.array(data)

        models = ["SurpriseSlopeOneModel",
                  "SurpriseSvdModel",
                  "SurpriseSvdPPModel",
                  "SurpriseNMF",
                  "SurpriseKNNBasic",
                  "SurpriseKNNWithMeans",
                  "SurpriseKNNBaseline",
                  "SurpriseCoClustering",
                  "SurpriseBaselineOnly"]

        for model in models:
            print("Start: {}".format(model))
            start_time = time.time()
            result = m.cross_validates_one_by_one(p, data, model)
            diff =  (time.time() - start_time)
            print("Time taken: {} {}s".format(model, diff))

    print("Script ended")

def save_to_pickle(data):
    pickle.dump( data, open( "data.p", "wb" ) )

if __name__ == "__main__":
   main()
   #save_to_pickle(data)
