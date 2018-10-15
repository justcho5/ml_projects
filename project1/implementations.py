import numpy as np
import csv

# Return type: Note that all functions should return: (w, loss), which is the last weight vector of the
# method, and the corresponding loss value (cost function). Note that while in previous labs you might have
# kept track of all encountered w for iterative methods, here we only want the last one.

# -*- coding: utf-8 -*-
"""some helper functions for project 1."""


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def compute_mse(y, tx, w):
    e = y - np.dot(tx, w)
    mse = (1 / 2) * np.mean(np.dot(e.T, e))
    return mse


def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return np.sqrt(2 * mse)


def compute_gradient(y, tx, w):
    N = y.shape[0]
    e = np.subtract(y, np.dot(tx, w))
    gradient = -(1 / (2 * N)) * np.dot((tx.T), (e))
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using gradient descent
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * (gradient)
        # store w and loss

    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using stochastic gradient descent
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):

            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(minibatch_y, minibatch_tx, w)

            w = w - gamma * (gradient)
        # store w and loss

    return (w, loss)


def least_squares(y, tx):
    # Least squares regression using normal equations
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a, b)
    # Here I used RMSE
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a + aI, b)
    # RMSE?
    loss = np.sqrt(2 * compute_mse(y, tx, w))
    return (w, loss)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_loss(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def logistic_regression_gradient(x, y, h):
    return (x.T @ (h - y)) / y.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent or SGD
    z = tx @ gamma
    h = sigmoid(z)

    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            gradient = logistic_regression_gradient(minibatch_tx, minibatch_y, h)
            loss = logistic_regression_loss(minibatch_y, h)
            w = w - gamma * (gradient)
            # store w and loss

    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent or SGD
    z = tx @ gamma
    h = sigmoid(z)

    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            gradient = logistic_regression_gradient(minibatch_tx, minibatch_y, h)

            w = w - gamma * (gradient)

            regularization = lambda_ / 2 * np.sum(w ** 2)
            loss = logistic_regression_loss(minibatch_y, h) + regularization

    return (w, loss)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    losses_tr = []
    losses_te = []
    for i, ind_te in enumerate(k_indices):
        y_te = y[ind_te]
        x_te = x[ind_te]

        ind_tr = np.vstack((k_indices[:i], k_indices[i + 1 :])).flatten()
        y_tr = y[ind_tr]
        x_tr = x[ind_tr]
        tx_tr = build_poly(x_tr, degree)
        tx_te = build_poly(x_te, degree)
        weights = ridge_regression(y_tr, tx_tr, lambda_)

        losses_tr.append(compute_rmse(y_tr, tx_tr, weights))
        losses_te.append(compute_rmse(y_te, tx_te, weights))

    loss_tr = sum(losses_tr) / len(losses_tr)
    loss_te = sum(losses_te) / len(losses_te)
    return loss_tr, loss_te


# Load the data and return y, x, and ids
train_datapath = "data/train.csv"
test_datapath = "data/test.csv"
y_tr, x_tr, ids_tr = load_csv_data(train_datapath)
y_te, x_te, ids_te = load_csv_data(test_datapath)

# Get the weights
weights = least_squares_GD(y_tr, tx, initial_w, max_iters, gamma)
