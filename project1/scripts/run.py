from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)

from helpers import (
    load_csv_data,
    predict_labels,
    create_csv_submission
)

from cross_val import (
    cross_validation,
    build_k_indices
)

import numpy as np

def predict_and_generate_file(weights):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, '../data/output.csv')

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


print("Number of NaNs for each feature", np.isnan(x_tr).sum(axis=0) / x_tr.shape[0])
# Number of NaNs for each feature [ 0.152456  0.        0.        0.        0.709828  0.709828  0.709828  0.
#   0.        0.        0.        0.        0.709828  0.        0.        0.
#   0.        0.        0.        0.        0.        0.        0.        0.399652
#   0.399652  0.399652  0.709828  0.709828  0.709828  0.      ]


# print("original", x_tr.shape)
# print("nans", x_tr[np.isnan(x_tr).any(axis=1)].shape)

# Standardize features
#x_tr = (x_tr - np.nanmean(x_tr, axis=0)) / np.nanstd(x_tr, axis=0)

# Remove rows which have one NaN ( just for test .. )
y_tr = y_tr[~np.isnan(x_tr).any(axis=1)]
ids_tr = ids_tr[~np.isnan(x_tr).any(axis=1)]
x_tr = x_tr[~np.isnan(x_tr).any(axis=1)]

print("Do least square with ", len(x_tr), " rows")
lambda_ = 2.27584592607e-05
initial_w = np.random.rand(30, 1)
# print(initial_w)
max_iters = 100
gamma = 1 / max_iters

# Get the weights
lr_weights, lr_loss = logistic_regression(y_tr, x_tr, np.zeros(30), max_iters, gamma)
print("Loss Logistic Regression: ", lr_loss)

reg_lr_weights, reg_lr_loss = reg_logistic_regression(y_tr, x_tr, lambda_, np.zeros(30), max_iters, gamma)
print("Loss Reg. Logistic Regression: ", reg_lr_loss)

weights_ridge, loss_ridge = ridge_regression(y_tr, x_tr, lambda_)
print("Loss Ridge Reg:", loss_ridge)

weights_ls_gd, loss_ls_gd = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
print("Loss least square GD: ", loss_ls_gd)

weights_ls_sgd, loss_ls_sgd = least_squares_SGD(y_tr, x_tr, initial_w, max_iters, gamma)
print("Loss least square SGD: ", loss_ls_sgd)

#k_indicies = build_k_indices(y_tr, 4, 1)
#loss_tr, loss_te = cross_validation(y_tr, x_tr, k_indicies, 2, lambda_, 1)
#print("Loss: ", loss_tr, loss_te)

#predict_and_generate_file(weights_ls_gd)
