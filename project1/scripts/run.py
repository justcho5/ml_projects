from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from helpers import load_csv_data

# Load the data and return y, x, and ids
train_datapath = "../data/train.csv"
test_datapath = "../data/test.csv"
y_tr, x_tr, ids_tr = load_csv_data(train_datapath)
y_te, x_te, ids_te = load_csv_data(test_datapath)

lambda_ = 2.27584592607e-05
# Get the weights
# weights = least_squares_GD(y_tr, tx, initial_w, max_iters, gamma)
ls_weights, ls_loss = least_squares(y_tr, x_tr)
rr_weights, rr_loss = ridge_regression(y_tr, x_tr, lambda_)
# lr_weights, lr_loss = logistic_regression()
# rlr_weights, rlr_loss
print(ls_loss, rr_loss)
