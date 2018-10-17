from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
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


def predict_and_generate_file(weights):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


# Load the data and return y, x, and ids
train_datapath = "../data/train.csv"
test_datapath = "../data/test.csv"

print("Load CSV file")
y_tr, x_tr, ids_tr = load_csv_data(train_datapath, sub_sample=True)
y_te, x_te, ids_te = load_csv_data(test_datapath)
print(x_tr)

# Preprocessing
print("Pre process: {} rows ".format(len(y_tr)))

# Replace -999 by NaN
x_tr[x_tr == -999] = np.nan
x_te[x_te == -999] = np.nan

# x_tr : original features
# x_tr1 : removes all features with NaNs. Resulting shape (250000, 19)
# x_tr2 : removes all features with > 70% NaNs and replaces the rest of the Nans with the feature mean (250000, 23)

x_tr1, x_te1 = remove_features(x_tr, x_te)
x_tr2, x_te2 = remove_features(x_tr, x_te, threshold=0.7, replace_with_mean=True)

x_tr1, x_te1 = standardize_features(x_tr1, x_te1)
x_tr2, x_te2 = standardize_features(x_tr2, x_te2)

print("Do least square with ", len(x_tr), " rows")
lambda_ = 2.27584592607e-05
initial_w = np.random.rand(30, 1)
# print(initial_w)
max_iters = 100
gamma = 1 / max_iters
# Get the weights

# lsgd_weights, lsgd_loss = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
# ls_weights, ls_loss = least_squares(y_tr, x_tr)

# lr_weights, lr_loss = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
# print("Loss LR: ", lr_loss)

weights_ridge, loss_ridge = ridge_regression(y_tr, x_tr, lambda_)
print("Loss Ridge Reg:", loss_ridge)

weights_ls_gd, loss_ls_gd = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
print("Loss least square GD: ", loss_ls_gd)

weights_ls_sgd, loss_ls_sgd = least_squares_SGD(y_tr, x_tr, initial_w, max_iters, gamma)
print("Loss least square SGD: ", loss_ls_gd)

# k_indicies = build_k_indices(y_tr, 4, 1)
# loss_tr, loss_te = cross_validation(y_tr, x_tr, k_indicies, 2, lambda_, 1)
# print("Loss: ", loss_tr, loss_te)

# predict_and_generate_file(weights_ls_gd)
