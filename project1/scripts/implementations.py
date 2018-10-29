import numpy as np

# Return type: Note that all functions should return: (w, loss), which is the last weight vector of the
# method, and the corresponding loss value (cost function). Note that while in previous labs you might have
# kept track of all encountered w for iterative methods, here we only want the last one.

# -*- coding: utf-8 -*-
"""some helper functions for project 1."""



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


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree + 1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def compute_mse(y, tx, w):
    e = y - (tx @ w)
    mse = (1 / 2) * np.mean(e.T @ e)
    return mse


def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return np.sqrt(2 * mse)


def compute_gradient(y, tx, w):
    N = y.shape[0]
    e = y - (tx @ w)
    gradient = -(1 / N) * (tx.T @ e)
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using gradient descent
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        # store w and loss

    loss = compute_rmse(y, tx, w)
    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Linear regression using stochastic gradient descent
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_rmse(minibatch_y, minibatch_tx, w)

            w = w - gamma * (gradient)
        # store w and loss

    return (w, loss)


def least_squares(y, tx):
    # Least squares regression using normal equations
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)

    loss = compute_rmse(y, tx, w)
    return (w, loss)


def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a + aI, b)

    loss = compute_rmse(y, tx, w)
    return (w, loss)

## Logistic regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_loss(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def logistic_regression_gradient(x, y, h):
    a = h - y
    r = x.T @ a
    return r / y.shape[0]


def reg_log_regression_gradient(x, y, h, w, lambda_):
    grad = logistic_regression_gradient(x, y, h)
    grad[1:] = grad[1:] + (lambda_ / y.shape[0]) * w[1:]
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent or SGD
    z = tx @ initial_w
    h = sigmoid(z)

    w = initial_w

    for n_iter in range(max_iters):
        gradient = logistic_regression_gradient(tx, y, h)
        w = w - gamma * (gradient)

    loss = compute_rmse(y, tx, w)
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent or SGD
    z = tx @ initial_w
    h = sigmoid(z)

    w = initial_w

    for n_iter in range(max_iters):
        gradient = reg_log_regression_gradient(tx, y, h, w, lambda_)
        w = w - gamma * (gradient)

    loss = compute_rmse(y, tx, w)
    return (w, loss)
