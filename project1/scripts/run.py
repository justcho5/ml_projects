from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)


# Load the data and return y, x, and ids
train_datapath = "data/train.csv"
test_datapath = "data/test.csv"
y_tr, x_tr, ids_tr = load_csv_data(train_datapath)
y_te, x_te, ids_te = load_csv_data(test_datapath)

# Get the weights
# weights = least_squares_GD(y_tr, tx, initial_w, max_iters, gamma)
weights = least_squares(y_tr, x_tr)
print(weights)
