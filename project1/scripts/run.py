from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    build_poly
)

from helpers import load_csv_data, predict_labels, create_csv_submission

from cross_val import cross_validation, build_k_indices

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

    return x_tr, x_te


def standardize_features(x_tr, x_te):
    mean_tr = np.mean(x_tr, axis=0)
    std_tr = np.std(x_tr, axis=0)
    x_tr = (x_tr - mean_tr) / std_tr
    x_te = (x_te - mean_tr) / std_tr

    #     print("First five train rows standardized: ", x_tr)
    #     print("First five test rows standardized: ", x_te)

    return x_tr, x_te


def predict_and_generate_file(weights, x_te):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


# Load the data and return y, x, and ids
train_datapath = "../data/train.csv"
test_datapath = "../data/test.csv"

print("Load CSV file")
y_tr, x_tr, ids_tr = load_csv_data(train_datapath, sub_sample=False)
y_te, x_te, ids_te = load_csv_data(test_datapath)
print(x_te)

# Preprocessing
print("Pre process: {} rows ".format(len(y_tr)))

# Replace -999 by NaN
x_tr[x_tr == -999] = np.nan

# x_tr : original features
# x_tr1 : removes all features with NaNs. Resulting shape (250000, 19)
# x_tr2 : removes all features with > 70% NaNs and replaces the rest of the Nans with the feature mean (250000, 23)

x_tr1, x_te1 = remove_features(x_tr, x_te)
x_tr2, x_te2 = remove_features(x_tr, x_te, threshold=0.7, replace_with_mean=True)

x_tr1, x_te1 = standardize_features(x_tr1, x_te1)
x_tr2, x_te2 = standardize_features(x_tr2, x_te2)

x_tr = x_tr1
x_te = x_te1

def try_different_model(x_tr, x_te):
    lambda_ = 2.27584592607e-05
    initial_w = np.random.rand(x_tr.shape[1], 1)
    max_iters = 100
    gamma = 1 / max_iters

    # Get the weights
    lr_weights, lr_loss = logistic_regression(y_tr, x_tr, np.zeros(x_tr.shape[1]), max_iters, gamma)
    print("Loss Logistic Regression: ", lr_loss)

    reg_lr_weights, reg_lr_loss = reg_logistic_regression(y_tr, x_tr, lambda_, np.zeros(x_tr.shape[1]), max_iters, gamma)
    print("Loss Reg. Logistic Regression: ", reg_lr_loss)

    x_tr = build_poly(x_tr, 2)
    x_te = build_poly(x_te, 2)
    weights_ridge, loss_ridge = ridge_regression(y_tr, x_tr, lambda_)
    print("Loss Ridge Reg:", loss_ridge)

    weights_ls, loss_ls = least_squares(y_tr, x_tr)
    print("Loss least square: ", loss_ls)

    #weights_ls_gd, loss_ls_gd = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
    #print("Loss least square GD: ", loss_ls_gd)

    #weights_ls_sgd, loss_ls_sgd = least_squares_SGD(y_tr, x_tr, initial_w, max_iters, gamma)
    #print("Loss least square SGD: ", loss_ls_sgd)

    # for degree in range(1, 5):
    #     k_indices = build_k_indices(y_tr, 10, 1)
    #     fun = lambda y, x: least_squares(y, x)
    #     loss_tr, loss_te = cross_validation(y_tr, x_tr, k_indices, degree, fun)
    #     print("CrossVal. Loss Logistic Regression: ", loss_tr, loss_te, " degree=", degree)
    #
    #     fun = lambda y, x: ridge_regression(y, x, lambda_)
    #     loss_tr, loss_te = cross_validation(y_tr, x_tr, k_indices, degree, fun)
    #     print("CrossVal. Loss Ridge Regression: ", loss_tr, loss_te, " degree=", degree)

try_different_model(x_tr, x_te)
#predict_and_generate_file(weights_ridge, x_te)

def run_different_degree():
    values = []

    for degree in range(1, 10):
        k_indices = build_k_indices(y_tr, 20, 1)
        fun = lambda y, x: least_squares(y, x)
        loss_tr, loss_te = cross_validation(y_tr, x_tr, k_indices, degree, fun)
        print("CrossVal. Loss Logistic Regression: ", loss_tr, loss_te, " degree=", degree)
        values.append((degree, loss_tr, loss_te))

    return values


