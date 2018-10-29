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
    """
    :returns the modified training and test dat
    """

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


def subset_training(degree, lambda_, y_tr, x_tr, y_te, x_te):
    """
    :return: the subset tx and weights
    """

    # Partition the data based on Jet Number
    y_tests, x_tests = subset(y_te, x_te)
    y_trains, x_trains = subset(y_tr, x_tr)

    subset_weights = []
    subset_tx_te = []

    # Train with each partition. Thus a different set of weights depending on the
    # Jet number.
    test_y_x = zip(y_tests, x_tests)
    train_y_x = zip(y_trains, x_trains)

    for ind, ((y_training, x_training), (y_test, x_test)) in enumerate(zip(train_y_x, test_y_x)):
        tx_tr, tx_te = standardize_features(x_training, x_test)
        tx_tr = build_poly(tx_tr, degree)
        tx_te = build_poly(tx_te, degree)
        tx_tr, tx_te = standardize_features(tx_tr, tx_te)

        # Train the model
        weights, _ = ridge_regression(y_training, tx_tr, lambda_)
        subset_weights.append(weights)
        subset_tx_te.append(tx_te)

    return subset_tx_te, subset_weights


def cross_validation(y, x, k_indices, lambda_, degree):
    """
    :return: the f1 scores for all k-folds
    """
    f1_scores = []

    # for each k-fold
    for i, index_test in enumerate(k_indices):

        # Split the training set into train and test sets for cross-validation.
        y_test = y[index_test]
        x_test = x[index_test]

        index_training = np.vstack((k_indices[:i], k_indices[i + 1:])).flatten()
        y_training = y[index_training]
        x_training = x[index_training]

        subset_tx_te, subset_weights = subset_training(degree, lambda_,
                                                       y_training, x_training,
                                                       y_test, x_test)

        # Build the predictions matrix based on jet number
        predictions_y = np.zeros(y_test.shape)
        for ind, test in enumerate(zip(subset_weights, subset_tx_te)):

            if (ind == 0) | (ind == 1):
                mask = x_test[:, 22] == ind
            else:
                mask = (x_test[:, 22] == ind) | (x_test[:, 22] == ind + 1)

            labels = predict_labels(test[0], test[1])
            predictions_y[mask] = labels

        # calculated the F1 score for the current k-fold
        f1 = f1_score(y_test, predictions_y)
        f1_scores.append(f1)

    return f1_scores


def predict(y_tr, x_tr, y_te, x_te, ids_te, degree=9, lambda_=0.0001):
    """
    Creates the output.csv file.

    :param y_tr:  the training y
    :param x_tr:  the training x
    :param y_te:  the test y
    :param x_te:  the text x
    :param ids_te: the ids to output
    :param degree:  the degree to use
    :param lambda_: the lambda to use
    """

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

    create_csv_submission(ids_te, predictions_y, "../data/output.csv")

def predict_and_generate_file(weights, x_te, ids_te):
    """
    Generates the prediction file with given weights
    """
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


def read_file():
    """
    Helper class to read the file and return the test and training data.
    """
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
    """
    Run training for given configuration
    """
    degree, lambda_, y_training, x_training = input

    print("Start degree {} lambda {}".format(degree, lambda_))

    k_indices = build_k_indices(y_training, 10)
    f1_scores = cross_validation(y_training, x_training, k_indices, lambda_, degree)

    model = Model(degree, lambda_, f1_scores)
    model.print()

    return model


def do_training_for_configuration(degrees, lambdas):
    """
    Test for all given degrees and lambdas the model and returns
    all models and the best model
    :param degrees: list of degrees to test
    :param lambdas:  list of lambdas to test
    :return:  the list of models with the result
    """
    np.random.seed(1)

    (y_tr, x_tr, ids_tr), _ = read_file()

    # This is the set of configuration we want to try
    arguments = []
    for deg in degrees:
        for lambda_ in lambdas:
            arguments.append((deg, lambda_, y_tr, x_tr))

    # Run all configuration in a pool
    with Pool(8) as pool:
        all_models = pool.map(train_for_configuration, arguments)

        best_model = max(all_models, key=lambda each: each.average_score())

        return (all_models, best_model)


def do_training():
    """
    Does the training with the list of models and lambdas we have tested.
    """
    default_range = range(1, 20)
    default_lambdas = np.logspace(-20, -10, 20)
    return do_training_for_configuration(default_range, default_lambdas)


def predict_with_best_model():
    (y_tr, x_tr, ids_tr), (y_te, x_te, ids_te) = read_file()

    predict(y_tr, x_tr, y_te, x_te, ids_te, degree=14, lambda_=1e-15)


def main():
    predict_with_best_model()
    #do_training()


if __name__ == "__main__":
    main()

