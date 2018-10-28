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

from cross_val import build_k_indices, f1_score
from multiprocessing import Pool

import numpy as np


import matplotlib.pyplot as plt




def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def cross_validation_demo(y, x, k_indices, degree):
    degree = degree
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for lambda_ in lambdas:
        loss_tr, loss_te, _ = cross_validation(y, x, k_indices, lambda_, degree)
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
    # ***************************************************
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)




def standardize_features(x_tr, x_te):
    # Skip first column
    mean_tr = np.mean(x_tr[:, 1:], axis=0)
    std_tr = np.std(x_tr[:, 1:], axis=0)
    # print(std_tr)
    # print(std_tr.shape)
    i = np.where(std_tr == 0.0)
    x_tr[:, 1:] = (x_tr[:, 1:] - mean_tr) / std_tr
    x_te[:, 1:] = (x_te[:, 1:] - mean_tr) / std_tr

    return x_tr, x_te


def subset(y, x):
    x_tr[x_tr == -999] = -10
    x_te[x_te == -999] = -10


    mask0 = x[:, 22] == 0
    mask1 = x[:, 22] == 1
    mask23 = (x[:, 22] == 2) | (x[:, 22] == 3)

    y_0 = y[mask0]
    y_1 = y[mask1]
    y_23 = y[mask23]

    x_0 = x[mask0]
    x_1 = x[mask1]
    x_23 = x[mask23]

    invalid0 = [0, 4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    invalid1 = [0, 4, 5, 6, 12, 22, 26, 27, 28]
    invalid23 = [0, 22]

    invalid00 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    invalid01 = [4, 5, 6, 12, 22, 26, 27, 28]
    invalid023 = [22]

    x0 = np.delete(x_0, invalid0, axis=1)
    x1 = np.delete(x_1, invalid1, axis=1)
    x23 = np.delete(x_23, invalid23, axis =1)

    x00 = np.delete(x_0, invalid00, axis=1)
    x01 = np.delete(x_1, invalid01, axis=1)
    x023 = np.delete(x_23, invalid023, axis=1)

    return [y_0, y_1, y_23], [x0, x1, x23]


def compute_rmse(y, tx, w):
    e = y - np.dot(tx, w)
    mse = (1 / 2) * np.mean(np.dot(e.T, e))
    return np.sqrt(2 * mse)


def cross_validation(y, x, k_indices, lambda_, degree):
    """return the loss of ridge regression."""

    losses_tr = []
    losses_te = []
    f1_scores = []

    for i, ind_te in enumerate(k_indices):
        y_te = y[ind_te]
        x_te = x[ind_te]

        ind_tr = np.vstack((k_indices[:i], k_indices[i + 1 :])).flatten()
        y_tr = y[ind_tr]
        x_tr = x[ind_tr]

        y_tests, x_tests = subset(y_te, x_te)
        y_trains, x_trains = subset(y_tr, x_tr)
        error_tr = np.zeros(y_tr.shape)
        error_te = np.zeros(y_te.shape)
        subset_weights = []
        subset_tx_te = []
        for ind, (trains, tests) in enumerate(
            zip(zip(y_trains, x_trains), zip(y_tests, x_tests))
        ):
            if (ind == 0) | (ind == 1):
                mask_tr = x_tr[:, 22] == ind
                mask_te = x_te[:, 22] == ind
            else:
                mask_tr = (x_tr[:, 22] == ind) |(x_tr[:, 22] == ind+1)
                mask_te = (x_te[:, 22] == ind) |(x_te[:, 22] == ind+1)
            tx_tr = build_poly(trains[1], degree)
            tx_te = build_poly(tests[1], degree)
            # skip the jet column

            tx_tr, tx_te = standardize_features(tx_tr, tx_te)
            weights, _ = ridge_regression(trains[0], tx_tr, lambda_)
            subset_weights.append(weights)
            subset_tx_te.append(tx_te)

            # print(error_tr.shape, mask_tr.shape)
            error_tr[mask_tr] = trains[0] - np.dot(tx_tr, weights)
            error_te[mask_te] = tests[0] - np.dot(tx_te, weights)
        losses_tr.append(np.mean(np.dot(error_tr.T, error_tr)))
        losses_te.append(np.mean(np.dot(error_te.T, error_te)))
        # calculated how many data-points are correctly predicted
        # This is different from the loss of the text set, since at the end we're
        # only interested on the prediction
        predictions_y = np.zeros(y_te.shape)
        for ind, test in enumerate(zip(subset_weights, subset_tx_te)):

            if (ind == 0) | (ind == 1):
                mask = x_te[:, 22] == ind
            else:
                mask = (x_te[:, 22] == ind) |(x_te[:, 22] == ind+1)
            labels = predict_labels(test[0], test[1])
            predictions_y[mask] = labels
        f1 = f1_score(y_te, predictions_y)
        f1_scores.append(f1)

    avg_loss_tr = np.average(losses_tr)
    avg_loss_te = np.average(losses_te)
    avg_f1_score = np.average(f1_scores)

    return avg_loss_tr, avg_loss_te, avg_f1_score

def model(y_tr, x_tr, y_te, x_te, ids_te, degree = 9, lambda_ = 0.0001):
    y_tests, x_tests = subset(y_te, x_te)
    y_trains, x_trains = subset(y_tr, x_tr)
    error_tr = np.zeros(y_tr.shape)
    error_te = np.zeros(y_te.shape)
    subset_weights = []
    subset_tx_te = []
    for ind, (trains, tests) in enumerate(
            zip(zip(y_trains, x_trains), zip(y_tests, x_tests))
    ):
        if (ind == 0) | (ind == 1):
            mask_tr = x_tr[:, 22] == ind
            mask_te = x_te[:, 22] == ind
        else:
            mask_tr = (x_tr[:, 22] == ind) | (x_tr[:, 22] == ind + 1)
            mask_te = (x_te[:, 22] == ind) | (x_te[:, 22] == ind + 1)
        tx_tr = build_poly(trains[1], degree)
        tx_te = build_poly(tests[1], degree)
        # skip the jet column

        tx_tr, tx_te = standardize_features(tx_tr, tx_te)
        weights, _ = ridge_regression(trains[0], tx_tr, lambda_)
        subset_weights.append(weights)
        subset_tx_te.append(tx_te)

    # calculated how many data-points are correctly predicted
    # This is different from the loss of the text set, since at the end we're
    # only interested on the prediction
    predictions_y = np.zeros(y_te.shape)
    for ind, test in enumerate(zip(subset_weights, subset_tx_te)):

        if (ind == 0) | (ind == 1):
            mask = x_te[:, 22] == ind
        else:
            mask = (x_te[:, 22] == ind) | (x_te[:, 22] == ind + 1)
        labels = predict_labels(test[0], test[1])
        predictions_y[mask] = labels
    create_csv_submission(ids_te, predictions_y, "../data/output_mult_models.csv")

def feat_perc_nan(x_tr):
    nan_ind = np.where(np.isnan(x_tr).sum(axis=0) > 0)
    feat = np.isnan(x_tr).sum(axis=0) / x_tr.shape[0]
    print("Features where NaNs exist: ", nan_ind)
    print(feat[nan_ind])


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def predict_and_generate_file(weights, x_te, ids_te):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


# split dataset based on jetnum number

train_datapath = "../data/train.csv"
test_datapath = "../data/test.csv"

print("Load CSV file")
y_tr, x_tr, ids_tr = load_csv_data(train_datapath, sub_sample=False)
y_te, x_te, ids_te = load_csv_data(test_datapath)

# x_tr[x_tr == -999] = np.nan
# x_te[x_te == -999] = np.nan
#
# # index 22 jet num
# mask0 = x_tr[:, 22] == 0
# mask1 = x_tr[:, 22] == 1
# mask2 = x_tr[:, 22] == 2
# mask3 = x_tr[:, 22] == 3
# x_tr0 = x_tr[mask0]
# x_tr1 = x_tr[mask1]
# x_tr2 = x_tr[mask2]
# x_tr3 = x_tr[mask3]
#
# y_tr0 = y_tr[mask0]
# y_tr1 = y_tr[mask1]
# y_tr2 = y_tr[mask2]
# y_tr3 = y_tr[mask3]
# print("Jetnum is 0", x_tr0.shape)
# print("Jetnum is 1", x_tr1.shape)
# print("Jetnum is 2", x_tr2.shape)
# print("Jetnum is 3", x_tr3.shape)
#
# feat_perc_nan(x_tr0)
# feat_perc_nan(x_tr1)
# feat_perc_nan(x_tr2)
# feat_perc_nan(x_tr3)

# invalid0 = [ 0,  4,  5,  6, 12, 23, 24, 25, 26, 27, 28]
# invalid1 = [ 0,  4,  5,  6, 12, 26, 27, 28]

# x_tr0 = np.delete(x_tr0, invalid0, axis=1)
# x_tr1 = np.delete(x_tr1, invalid0, axis=1)

# # x_tr0[np.isnan(x_tr0)] = 0
# # x_tr1[np.isnan(x_tr1)] = 1
# # x_tr2[np.isnan(x_tr2)] = 2
# # x_tr3[np.isnan(x_tr3)] = 3
#
np.random.seed(1)
k_indices = build_k_indices(y_tr, 10)
lambdas = np.logspace(-10,0,30)
#
for deg in range(14,15):
    for lambda_ in lambdas:

        degree = deg
        avg_loss_tr, avg_loss_te, avg_f1 = cross_validation(
            y_tr, x_tr, k_indices, lambda_, degree
        )
        print("Degree {degree}, lambda {lambda_}, Average loss training: ".format(degree = degree, lambda_ = lambda_), avg_loss_tr)
        print("Degree {degree}, lambda {lambda_}, Average loss test: ".format(degree = degree, lambda_ = lambda_),avg_loss_te)
        print("Degree {degree}, lambda {lambda_}, Average f1_score: ".format(degree = degree, lambda_ = lambda_),avg_f1)

# Jetnum is 0 (404, 30)
# Jetnum is 1 (319, 30)
# Jetnum is 2 (173, 30)
# Jetnum is 3 (103, 30)
# Features where NaNs exist:  (array([ 0,  4,  5,  6, 12, 23, 24, 25, 26, 27, 28]),)
# [ 0.22277228  1.          1.          1.          1.          1.          1.
#   1.          1.          1.          1.        ]
# Features where NaNs exist:  (array([ 0,  4,  5,  6, 12, 26, 27, 28]),)
# [ 0.10031348  1.          1.          1.          1.          1.          1.
#   1.        ]
# Features where NaNs exist:  (array([0]),)
# [ 0.05202312]
# Features where NaNs exist:  (array([0]),)
# [ 0.03883495]
# cross_validation_demo(y_tr, x_tr, k_indices, 8)

# model(y_tr, x_tr, y_te, x_te, ids_te, degree = 14, lambda_= 0.0001)