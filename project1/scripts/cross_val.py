import numpy as np

from implementations import (
    build_poly,
    ridge_regression,
    compute_rmse
)

from helpers import predict_labels


def precision(label, prediction):
    """
    Return recall (tp / (tp + fn))
    """

    assert len(prediction) == len(label), "pred and label different size"

    tp_ind = np.where((prediction == label) & (label == 1))
    tp = len(tp_ind[0])
    fp_ind = np.where((prediction != label) & (prediction == 1))
    fp = len(fp_ind[0])
    return 1 - (tp / (tp + fp))


def recall(label, prediction):
    """
    Return recall (tp / (tp + fn))
    """

    assert len(prediction) == len(label), "pred and label different size"

    tp_ind = np.where((prediction == label) & (label == 1))
    tp = len(tp_ind[0])
    fn_ind = np.where((prediction != label) & (label == 1))
    fn = len(fn_ind[0])
    return 1 - (tp / (tp + fn))


def f1_score(label, prediction):
    assert len(prediction) == len(label), "pred and label different size"

    p = precision(label, prediction)
    r = recall(label, prediction)
    return 1 - (2 * p * r / (p + r))

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

        # Predict labels using the current weights.
        predictions_y = predict_labels(weights, tx_te)

        ## http://kawahara.ca/how-to-compute-truefalse-positives-and-truefalse-negatives-in-python-for-binary-classification-problems/
        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(predictions_y == -1, y_te == -1))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(predictions_y == 1, y_te == 1))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(predictions_y == -1, y_te == 1))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(predictions_y == 1, y_te == -1))

        A =  (TP * TN ) - (FP * FN)
        X = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
        MCC = A / X

        accuracy = f1_score(y_te, predictions_y)
        #accuracy = false_positive / false_negative
        #accuracy = MCC
        accuracies.append(accuracy)



    return losses_tr, losses_te, accuracies
