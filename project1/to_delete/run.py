from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    build_poly,
)

from helpers import load_csv_data, predict_labels, create_csv_submission

from cross_val import cross_validation, build_k_indices
from multiprocessing import Pool

import numpy as np


def remove_features(x_tr, x_te, threshold=0, replace_with_mean=False):
    #     print("First five train rows before:\n", x_tr[:5], "\n")
    #     print("First five test rows before:\n", x_te[:5], "\n")

    mask = np.isnan(x_tr).sum(axis=0) / x_tr.shape[0] > threshold
    x_te = x_te[:, ~mask]
    x_tr = x_tr[:, ~mask]
    if replace_with_mean:
        col_mean = np.nanmean(x_tr, axis=0)
        #         print("COLUMN MEAN:\n", col_mean, "\n")
        nan_inds_tr = np.where(np.isnan(x_tr))
        nan_inds_te = np.where(np.isnan(x_te))

        x_te[nan_inds_te] = np.take(col_mean, nan_inds_te[1])
        x_tr[nan_inds_tr] = np.take(col_mean, nan_inds_tr[1])
    #     print("First five train rows after:\n", x_tr[:5], "\n")
    #     print("First five test rows after:\n", x_te[:5], "\n")
    #     print("Number of NaNs per training feature:\n", np.isnan(x_tr).sum(axis=0), "\n")
    #     print("Number of NaNs per test feature:\n", np.isnan(x_te).sum(axis=0), "\n")

    # 20 PRI_met_phi
    # 18 PRI_lep_phi
    # 15 PRI_tau_phi
    # 23 PRI_jet_leading_phi
    # 26 PRI_jet_subleading_phi
    # 14 PRI_tau_eta
    # 17 PRI_lep_eta

    index = [20, 18, 15]
    x_te = np.delete(x_te, index, axis=1)
    x_tr = np.delete(x_tr, index, axis=1)

    return x_tr, x_te


def standardize_features(x_tr,x_te):
    #standardization with respect to the training set
    
    mean_tr = np.mean(x_tr[:, 1:], axis=0)
    std_tr = np.std(x_tr[:, 1:], axis=0)
    x_tr[:, 1:] = (x_tr[:, 1:] - mean_tr) / std_tr
    x_te[:, 1:] = (x_te[:, 1:] - mean_tr) / std_tr
    
    #normalization with respect to the training set 
    #min_tr = np.amin(x_tr[:, 1:], axis= 0 )
    #max_tr = np.amax(x_tr[:, 1:], axis= 0 )
    #x_tr[:, 1:] = (x_tr[:, 1:] - min_tr)/(max_tr - min_tr)
    #x_te[:, 1:] = (x_te[:, 1:] - min_tr)/(max_tr - min_tr)
    
    #     print("First five train rows standardized: ", x_tr)
    #     print("First five test rows standardized: ", x_te)
    return x_tr,x_te
    
    
def standardize_feature(x_tr):
    
    #standardization with respect to the training set
    mean_tr = np.mean(x_tr[:, 1:], axis=0)
    std_tr = np.std(x_tr[:, 1:], axis=0)
    x_tr[:, 1:] = (x_tr[:, 1:] - mean_tr) / std_tr
    
    #normalization 
    #min_tr = np.amin(x_tr[:, 1:], axis= 0 )
    #max_tr = np.amax(x_tr[:, 1:], axis= 0 )
    #x_tr[:, 1:] = (x_tr[:, 1:] - min_tr)/(max_tr - min_tr)
    return x_tr


def predict_and_generate_file(weights, x_te, ids_te):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction) 
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


def read_train_test():
    # Load the data and return y, x, and ids
    train_datapath = "../data/train.csv"
    test_datapath = "../data/test.csv"

    print("Load CSV file")
    y_tr, x_tr, ids_tr = load_csv_data(train_datapath, sub_sample=False)
    y_te, x_te, ids_te = load_csv_data(test_datapath)
    # print(x_te)

    # Replace -999 by 0
    x_tr[x_tr == -999] = 0
    x_te[x_te == -999] = 0

    # x_tr : original features
    # x_tr1 : removes all features with NaNs. Resulting shape (250000, 19)
    # x_tr2 : removes all features with > 70% NaNs and replaces the rest of the Nans with the feature mean (250000, 23)

    x_tr1, x_te1 = remove_features(x_tr, x_te)
    x_tr2, x_te2 = remove_features(x_tr, x_te, threshold=0.7, replace_with_mean=True)
    # x_tr3, x_te3 = np.sqrt(x_tr2, x_te2)

    x_tr1, x_te1 = standardize_features(x_tr1, x_te1)
    x_tr2, x_te2 = standardize_features(x_tr2, x_te2)

    print("Pre process: initial {}, after cleaning {}".format(x_tr.shape, x_tr2.shape))
    x_tr = x_tr2
    x_te = x_te2

    return x_tr, y_tr, x_te, y_te, ids_te


class Model:
    def __init__(self, name, degree):
        self.name = name
        self.degree = degree

    def run(self, y_training, x_training, k_indices, model_function):
        print(self.degree, "Run", self.name)
        (losses_training, losses_test, f1_scores) = cross_validation(
            y_training, x_training, k_indices, self.degree, model_function
        )

        self.loss_training = losses_training
        self.loss_test = losses_test
        self.f1_scores = f1_scores

        self.loss_training = np.average(losses_training)
        self.loss_test = np.average(losses_test)
        self.f1_score = np.average(f1_scores)

        # run it with the whole data set
        extended_training_set = build_poly(x_training, self.degree)
        extended_training_set = standardize_feature(extended_training_set)
        (weight, total_loss) = model_function(y_training, extended_training_set)
        self.weights = weight
     

    def print(self):
        print(
            self.degree,
            np.round(self.loss_training, 4),
            np.round(self.loss_test, 4),
            np.round(self.f1_score, 4),
            self.name,
        )


def try_different_models(x_training, y_training):
    print("Try different models")
    #lambdas= np.linspace(0.00001,0.00015,10)

    # Let's run this in a pool, so we can use all the available CPU Cores
    # 8 ist just a number which I've chosen ( no deeper meaning )
    with Pool(8) as pool:
        arguments = []
        
        for degree in range(1, 21):
            arguments.append((degree, x_training, y_training))
            
        # submit jobs to try all degrees
        best_model_for_degree = pool.map(try_all_models_for_degree, arguments)
        best_model = max(
            best_model_for_degree, key=lambda model: model.f1_score
        )
        
        return (best_model_for_degree, best_model)


def try_all_models_for_degree(degree_and_data):
    degree, x_training, y_training = degree_and_data
    print("Start Degree = ", degree)
    #print("StartLlambda = ", lambda_)

    lambda_ = 10e-10
    max_iters = 100**2
    gamma = 0.0005
    
    k_indices = build_k_indices(y_training, 10)
    
    '''
    #m_least_square = Model("Least Square", degree)
    #m_least_square.run(y_training, x_training, k_indices, least_squares)

    #model_function = lambda y, x: least_squares_GD(
    #    y, x, np.zeros(x.shape[1]), max_iters, gamma
    #)
    #m_least_square_gd = Model("Least Square GD", degree)
    #m_least_square_gd.run(y_training, x_training, k_indices, model_function)
    
    
    #model_function = lambda y, x: least_squares_SGD(
    #    y, x, np.zeros(x.shape[1]), max_iters, gamma
    #)
    #m_least_square_sgd = Model("Least Square SGD", degree)
    #m_least_square_sgd.run(y_training, x_training, k_indices, model_function)
    
    
    model_function = lambda y, x: logistic_regression(
        y, x, np.zeros(x.shape[1]), max_iters, gamma
    )
    m_logistic_regression = Model("Logistic Regression", degree)
    m_logistic_regression.run(y_training, x_training, k_indices, model_function)

    
    model_function = lambda y, x: reg_logistic_regression(
        y, x, lambda_, np.zeros(x.shape[1]), max_iters, gamma
    )
    m_reg_logistic_regression = Model("Reg. Logistic Regression", degree)
    m_reg_logistic_regression.run(y_training, x_training, k_indices, model_function)
    '''
    model_function = lambda y, x: ridge_regression(y, x, lambda_)
    m_ridge_regression = Model("Ridge Regression", degree)
    m_ridge_regression.run(y_training, x_training, k_indices, model_function)
    
    '''
    all_models = [
        m_least_square,
        m_logistic_regression,
        m_least_square_gd,
        m_least_square_sgd,
        m_reg_logistic_regression,
        m_ridge_regression,
    ]
    '''
    all_models = [m_ridge_regression
    ]

    # After all are done, print result
    for each_model in all_models:
        each_model.print()

    # get best model based on the loss
    best_model = max(all_models, key=lambda each: each.f1_score)
    return best_model


def run_all():
    np.random.seed(10)
    x_tr, y_tr, x_te, y_te, ids_te = read_train_test()
    x_tr, x_te = standardize_features(x_tr, x_te)

    (all, best_overall_model) = try_different_models(x_tr, y_tr)
    print("Best model is:")
    best_overall_model.print()

    print(best_overall_model.weights)

    extended_test_feature_te = build_poly(x_te, best_overall_model.degree)
    extended_test_feature_tr = build_poly(x_tr, best_overall_model.degree)
    extended_test_feature_tr, extended_test_feature_te = standardize_features(extended_test_feature_tr, extended_test_feature_te)
    predict_and_generate_file(best_overall_model.weights, extended_test_feature_te, ids_te)

    return all


#run_all()
    
    
