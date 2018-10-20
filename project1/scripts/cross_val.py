import numpy as np

from implementations import (
    build_poly,
    ridge_regression,
    compute_rmse
)

from helpers import predict_labels


def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)

    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, degree, model_function):
    """return the loss of ridge regression."""

    losses_tr = []
    losses_te = []
    accuracies = []

    for i, ind_te in enumerate(k_indices):
        y_te = y[ind_te]
        x_te = x[ind_te]

        ind_tr = np.vstack((k_indices[:i], k_indices[i + 1:])).flatten()
        y_tr = y[ind_tr]
        x_tr = x[ind_tr]

        tx_tr = build_poly(x_tr, degree)
        tx_te = build_poly(x_te, degree)
        weights, loss = model_function(y_tr, tx_tr)

        losses_tr.append(compute_rmse(y_tr, tx_tr, weights))
        losses_te.append(compute_rmse(y_te, tx_te, weights))

        # calculated how many data-points are correctly predicted
        # This is different from the loss of the text set, since at the end we're
        # only interested on the prediction
        predictions_y = predict_labels(weights, tx_te)
        diff = predictions_y - y_te
        accuracy = (len(diff) - np.count_nonzero(diff)) / len(diff)
        accuracies.append(accuracy)

    avg_loss_tr = np.average(losses_tr)
    avg_loss_te = np.average(losses_te)
    avg_accuracy = np.average(accuracies)

    return avg_loss_tr, avg_loss_te, avg_accuracy
