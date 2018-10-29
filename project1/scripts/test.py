from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    build_poly,
    compute_rmse,
)

from helpers import (
    load_csv_data,
    predict_labels,
    create_csv_submission
)

from multiprocessing import Pool
import numpy as np


def build_k_indices(y, k_fold):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)

    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def precision(y_training, y_prediction):
    """
    :returns recall (tp / (tp + fn))
    """
    true_positive = np.where((y_prediction == y_training) & (y_training == 1))
    false_positive = np.where((y_prediction != y_training) & (y_prediction == 1))

    len_tp = len(true_positive[0])
    len_fp = len(false_positive[0])

    return 1 - (len_tp / (len_tp + len_fp))


def recall(y_training, y_prediction):
    """
    :returns recall (tp / (tp + fn))
    """

    true_positive = np.where((y_prediction == y_training) & (y_training == 1))
    false_negative = np.where((y_prediction != y_training) & (y_training == 1))

    tp = len(true_positive[0])
    fn = len(false_negative[0])
    return 1 - (tp / (tp + fn))


def f1_score(y_training, y_prediction):
    """
    :returns the calculated f1 score for given y values
    """
    precision_value = precision(y_training, y_prediction)
    recall_value = recall(y_training, y_prediction)

    return 1 - (2 * precision_value * recall_value / (precision_value + recall_value))


def standardize_features(x_training, x_test):
    # Skip first column
    mean_tr = np.mean(x_training[:, 1:], axis=0)
    std_tr = np.std(x_training[:, 1:], axis=0)
    x_training[:, 1:] = (x_training[:, 1:] - mean_tr) / std_tr
    x_test[:, 1:] = (x_test[:, 1:] - mean_tr) / std_tr

    return x_training, x_test


def subset(y, x):
    """
    Partition the dataset into subsets based on the Jet Number (Feature 23).
    For each partition, drops feature 23 and features that are all NaNs, and replace
    the few remaining NaNs (are in the first column), with 0.
    """
    x[x == -999] = 0

    mask0 = x[:, 22] == 0
    mask1 = x[:, 22] == 1
    mask23 = (x[:, 22] == 2) | (x[:, 22] == 3)

    y_0 = y[mask0]
    y_1 = y[mask1]
    y_23 = y[mask23]

    x_0 = x[mask0]
    x_1 = x[mask1]
    x_23 = x[mask23]

    invalid0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    invalid1 = [4, 5, 6, 12, 22, 26, 27, 28]
    invalid23 = [22]

    x0 = np.delete(x_0, invalid0, axis=1)
    x1 = np.delete(x_1, invalid1, axis=1)
    x23 = np.delete(x_23, invalid23, axis=1)

    return [y_0, y_1, y_23], [x0, x1, x23]


def cross_validation(y, x, k_indices, lambda_, degree):
    """
    TODO KIRU
    :param y:
    :param x:
    :param k_indices:
    :param lambda_:
    :param degree:
    :return:
    """
    f1_scores = []

    for i, ind_te in enumerate(k_indices):
        # Split the training set into train and test sets for cross-validation.
        y_te = y[ind_te]
        x_te = x[ind_te]

        ind_tr = np.vstack((k_indices[:i], k_indices[i + 1:])).flatten()
        y_tr = y[ind_tr]
        x_tr = x[ind_tr]

        # Partition the data based on Jet Number
        y_tests, x_tests = subset(y_te, x_te)
        y_trains, x_trains = subset(y_tr, x_tr)

        error_tr = np.zeros(y_tr.shape)
        error_te = np.zeros(y_te.shape)
        subset_weights = []
        subset_tx_te = []

        # Train with each partition. Thus a different set of weights depending on the
        # Jet number.
        for ind, (trains, tests) in enumerate(
                zip(zip(y_trains, x_trains), zip(y_tests, x_tests))
        ):
            if (ind == 0) | (ind == 1):
                mask_tr = x_tr[:, 22] == ind
                mask_te = x_te[:, 22] == ind
            else:
                mask_tr = (x_tr[:, 22] == ind) | (x_tr[:, 22] == ind + 1)
                mask_te = (x_te[:, 22] == ind) | (x_te[:, 22] == ind + 1)

            tx_tr, tx_te = standardize_features(trains[1], tests[1])
            tx_tr = build_poly(tx_tr, degree)
            tx_te = build_poly(tx_te, degree)
            tx_tr, tx_te = standardize_features(tx_tr, tx_te)

            # Train the model
            weights, _ = ridge_regression(trains[0], tx_tr, lambda_)
            subset_weights.append(weights)
            subset_tx_te.append(tx_te)

            error_tr[mask_tr] = trains[0] - np.dot(tx_tr, weights)
            error_te[mask_te] = tests[0] - np.dot(tx_te, weights)

        # calculated how many data-points are correctly predicted
        # This is different from the loss of the test set, since at the end we're
        # only interested on the prediction

        # Build the predictions matrix based on jet number
        predictions_y = np.zeros(y_te.shape)
        for ind, test in enumerate(zip(subset_weights, subset_tx_te)):

            if (ind == 0) | (ind == 1):
                mask = x_te[:, 22] == ind
            else:
                mask = (x_te[:, 22] == ind) | (x_te[:, 22] == ind + 1)
            labels = predict_labels(test[0], test[1])
            predictions_y[mask] = labels
        f1 = f1_score(y_te, predictions_y)
        f1_scores.append(f1)

    return f1_scores


def model(y_tr, x_tr, y_te, x_te, ids_te, degree=9, lambda_=0.0001):
    y_tests, x_tests = subset(y_te, x_te)
    y_trains, x_trains = subset(y_tr, x_tr)

    subset_weights = []
    subset_tx_te = []
    for ind, (trains, tests) in enumerate(
            zip(zip(y_trains, x_trains), zip(y_tests, x_tests))
    ):
        tx_tr, tx_te = standardize_features(trains[1], tests[1])
        tx_tr = build_poly(tx_tr, degree)
        tx_te = build_poly(tx_te, degree)
        tx_tr, tx_te = standardize_features(tx_tr, tx_te)

        weights, _ = ridge_regression(trains[0], tx_tr, lambda_)

        subset_weights.append(weights)
        subset_tx_te.append(tx_te)

    # Build the Predictions matrix
    predictions_y = np.zeros(y_te.shape)
    for ind, test in enumerate(zip(subset_weights, subset_tx_te)):

        if (ind == 0) | (ind == 1):
            mask = x_te[:, 22] == ind
        else:
            mask = (x_te[:, 22] == ind) | (x_te[:, 22] == ind + 1)
        labels = predict_labels(test[0], test[1])
        predictions_y[mask] = labels
    create_csv_submission(ids_te, predictions_y, "../data/output_mult_models.csv")


def feat_perc_nan(x_tr):
    nan_ind = np.where(np.isnan(x_tr).sum(axis=0) > 0)
    feat = np.isnan(x_tr).sum(axis=0) / x_tr.shape[0]
    print("Features where NaNs exist: ", nan_ind)
    print(feat[nan_ind])


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def predict_and_generate_file(weights, x_te, ids_te):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


def read_file():
    train_datapath = "../data/train.csv"
    test_datapath = "../data/test.csv"

    print("Load CSV file")
    training_data = load_csv_data(train_datapath, sub_sample=False)
    test_data = load_csv_data(test_datapath)

    return training_data, test_data


class Model:
    """
    Represents a result from running with different configuration
    """

    def __init__(self, degree, lambda_, f1_scores):
        self.f1_scores = f1_scores
        self.lambda_ = lambda_
        self.degree = degree

    def average_score(self):
        return np.average(self.degree)

    def print(self):
        rounded_lambda = np.round(self.lambda_, 4),
        avg_score = np.round(np.average(self.f1_scores), 4)
        print("Degree {degree}, lambda {lambda_}, Average F1-Score: {avg_score}"
              .format(degree=self.degree,
                      lambda_=rounded_lambda,
                      avg_score=avg_score))


def train_for_configuration(input):
    degree, lambda_, y_training, x_training = input

    print("Start degree {} lambda {}".format(degree, lambda_))

    k_indices = build_k_indices(y_training, 10)
    f1_scores = cross_validation(y_training, x_training, k_indices, lambda_, degree)

    model = Model(degree, lambda_, f1_scores)
    model.print()

    return model


def do_training():
    np.random.seed(1)

    (y_tr, x_tr, ids_tr), (ignore_training_data) = read_file()

    # This is the set of configuration we want to try
    arguments = []
    for deg in range(1, 20):
        for lambda_ in np.logspace(-20, -1, 20):
            arguments.append((deg, lambda_, y_tr, x_tr))

    # Run all configuration in a pool
    with Pool(8) as pool:
        all_models = pool.map(train_for_configuration, arguments)
        best_model = max(all_models, key=lambda each: each.average_score())

        return (all_models, best_model)


def predict_with_best_model():
    (y_tr, x_tr, ids_tr), (y_te, x_te, ids_te) = read_file()

    model(y_tr, x_tr, y_te, x_te, ids_te, degree=14, lambda_=10e-15)


def main():
    # predict_with_best_model()
    do_training()


if __name__ == "__main__":
    main()
