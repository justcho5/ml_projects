from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from helpers import load_csv_data
import numpy as np

# Load the data and return y, x, and ids
train_datapath = "../data/small-train.csv"
test_datapath = "../data/small-test.csv"
y_tr, x_tr, ids_tr = load_csv_data(train_datapath)
y_te, x_te, ids_te = load_csv_data(test_datapath)
print(x_tr)
# Preprocessing

# Replace -999 by NaN
x_tr[x_tr == -999] = np.nan
# Standardize features
x_tr = (x_tr - np.nanmean(x_tr, axis=0)) / np.nanstd(x_tr, axis=0)

lambda_ = 2.27584592607e-05
initial_w = np.random.rand(30, 1)
# print(initial_w)
max_iters = 100
gamma = 1 / max_iters
# Get the weights
lsgd_weights, lsgd_loss = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
# ls_weights, ls_loss = least_squares(y_tr, x_tr)
# rr_weights, rr_loss = ridge_regression(y_tr, x_tr, lambda_)
# lr_weights, lr_loss = logistic_regression()
# rlr_weights, rlr_loss
print(lsgd_weights)
print(lsgd_loss)
