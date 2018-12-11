# -*- coding: utf-8 -*-
import pandas as pd
import surprise as s
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SlopeOne
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import CoClustering
from surprise import BaselineOnly
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

#from tqdm import tqdm_notebook
#tqdm = tqdm_notebook

#class Dataset:
    #def __init(self, full_data):
        #self.full_data = full_data

def to_surprise_datset(trainset, testset):
    df_data = pd.DataFrame(trainset, columns=['userID', 'itemID', 'rating'])
    reader = Reader(rating_scale=(1, 5))
    data_train = s.dataset.DatasetAutoFolds(df=df_data, reader=reader)
    return data_train

class SurpriseBaselineOnly:
    def __init__(self):
        self.name = 'BaselineOnly'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = BaselineOnly()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseCoClustering:
    def __init__(self):
        self.name = 'CoClustering'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = CoClustering()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseKNNBaseline:
    def __init__(self):
        self.name = 'KNNBaseline'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = KNNBaseline()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseKNNWithMeans:
    def __init__(self):
        self.name = 'KNNWithMeans'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = KNNWithMeans()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)

        self.algo = algo

class SurpriseKNNBasic:
    def __init__(self):
        self.name = 'KNNBasic'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = KNNBasic()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseNMF:
    def __init__(self):
        self.name = 'NMF'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = NMF()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseSvdPPModel:
    def __init__(self):
        self.name = 'Surprise SVDpp'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = SVDpp()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseSvdModel:
    def __init__(self):
        self.name = 'Surprise SVD'

    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = SVD()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

class SurpriseSlopeOneModel:
    def __init__(self):
        self.name = 'Surprise SlopeOne'
    
    def fit(self, trainset, testset):
        data_train = to_surprise_datset(trainset, testset)

        # train and test algorithm.
        algo = SlopeOne()

        algo.fit(data_train.build_full_trainset())
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        self.rmse = accuracy.rmse(predictions, verbose=False)
        self.algo = algo

def call_algo(i):
    trainset, testset = i
    print("Run a fold")

    models = [ 
        SurpriseSlopeOneModel(), 
        SurpriseSvdModel(),
        SurpriseSvdPPModel(),
        SurpriseNMF(),
        SurpriseKNNBasic(),
        SurpriseKNNWithMeans(),
        SurpriseKNNBaseline(),
        SurpriseCoClustering(),
        SurpriseBaselineOnly(),
    ]

    for m in tqdm(models, desc="One split"):
        print("run {}".format(m.name))
        m.fit(trainset, testset)

    return models

def cross_validate(pool, whole_data, is_parallel=True):
    kf = KFold(n_splits = 15)

    results = []
    splits = list(kf.split(whole_data))

    print("running CV")
    ## run the code sequentially or parallely
    if is_parallel:
        x = list(map(lambda x: (whole_data[x[0]], whole_data[x[1]]), splits))
        for result in tqdm(pool.imap(call_algo, x), total=len(x), desc="CV"):
            results.append(result)
    else:
        for train, test in tqdm(splits):
            trainset = whole_data[train]
            testset = whole_data[test]
            train_test = (trainset, testset)
            results.append(call_algo(train_test))

    for m in range(len(results[0])):
        for r in results:
            print(r[m].name, r[m].rmse)

    return results
