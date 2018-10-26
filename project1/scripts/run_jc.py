from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    build_poly,
)

from helpers import load_csv_data, predict_labels, create_csv_submission

from cross_val import build_k_indices
from multiprocessing import Pool

import numpy as np

from implementations import build_poly, ridge_regression, compute_rmse

from helpers import predict_labels


def change_features(
    x_tr, x_te, threshold=0, replace_with_mean=False, replace_with_zero=False
):

    # Input:
    # Original training and testing,
    # Remove features with greater than threshold of NaNs
    # Replace remaining Nans with mean or zeros

    # Output:
    # Transformed training and test feature sets, unstandardized

    mask = np.isnan(x_tr).sum(axis=0) / x_tr.shape[0] > threshold
    x_te = x_te[:, ~mask]
    x_tr = x_tr[:, ~mask]
    if replace_with_mean:
        col_mean = np.nanmean(x_tr, axis=0)
        nan_inds_tr = np.where(np.isnan(x_tr))
        nan_inds_te = np.where(np.isnan(x_te))

        x_te[nan_inds_te] = np.take(col_mean, nan_inds_te[1])
        x_tr[nan_inds_tr] = np.take(col_mean, nan_inds_tr[1])

    if replace_with_zero:
        x_tr = np.nan_to_num(x_tr)
        x_te = np.nan_to_num(x_te)
        
    # 20 PRI_met_phi
    # 18 PRI_lep_phi
    # 15 PRI_tau_phi
    # 23 PRI_jet_leading_phi
    # 26 PRI_jet_subleading_phi
    # 14 PRI_tau_eta
    # 17 PRI_lep_eta

    index = [20, 18, 15]
    x_te = np.delete(x_te, index, axis=1)
    x_tr = np.delete(x_tr, index, axis=1)

    return x_tr, x_te


def standardize_features_before(x_tr, x_te):
    # Skip first column
    mean_tr = np.mean(x_tr, axis=0)

    std_tr = np.std(x_tr, axis=0)
    x_tr = (x_tr - mean_tr) / std_tr
    x_te = (x_te - mean_tr) / std_tr

    return x_tr, x_te


def standardize_features(x_tr, x_te):
    # Skip first column
    mean_tr = np.mean(x_tr[:, 1:], axis=0)
    std_tr = np.std(x_tr[:, 1:], axis=0)
    print(std_tr)
    print(std_tr.shape)
    x_tr[:, 1:] = (x_tr[:, 1:] - mean_tr) / std_tr
    x_te[:, 1:] = (x_te[:, 1:] - mean_tr) / std_tr

    return x_tr, x_te


def combine(x_tr, x_te):
    array = np.ma.array(x_tr, mask=np.isnan(x_tr))
    corr = np.array(np.ma.corrcoef(array, rowvar=False, allow_masked=True))
    ind = np.argwhere(corr > 0.2)
    s = []
    result_tr = np.empty(x_tr.shape[0])
    result_te = np.empty(x_te.shape[0])

    for [i, j] in ind:
        if i != j:
            if (i, j) not in s:
                if (j, i) not in s:
                    s.append((i, j))
                    result_tr = np.c_[result_tr, x_tr[:, i] * x_tr[:, j]]
                    result_te = np.c_[result_te, x_te[:, i] * x_te[:, j]]
    return result_tr, result_te


def predict_and_generate_file(weights, x_te, ids_te):
    print("Predict for test data")
    y_prediction = predict_labels(weights, x_te)

    print("Predictions: ", y_prediction)
    print("Create submission file")
    create_csv_submission(ids_te, y_prediction, "../data/output.csv")


def read_train_test():
    # Load the data and return y, x, and ids
    train_datapath = "../data/train.csv"
    test_datapath = "../data/test.csv"

    print("Load CSV file")
    y_tr, x_tr, ids_tr = load_csv_data(train_datapath, sub_sample=False)
    y_te, x_te, ids_te = load_csv_data(test_datapath)
    # print(x_te)

    # Replace -999 by 0
    x_tr[x_tr == -999] = 0
    x_te[x_te == -999] = 0

<<<<<<< HEAD
=======
    # Dataset 1: Remove all columns with any NaNs
    x_tr1, x_te1 = change_features(x_tr, x_te)

    # Dataset 2: Replace all NaNs with mean
    x_tr2, x_te2 = change_features(x_tr, x_te, threshold=1, replace_with_mean=True)

    # Dataset 3: Replace all NaNs with zero
    x_tr3, x_te3 = change_features(x_tr, x_te, threshold=1, replace_with_zero=True)

    # Dataset 4: Remove all columns with >70% NaNs and replace the rest with mean
    x_tr4, x_te4 = change_features(x_tr, x_te, threshold=0.7, replace_with_mean=True)

    # Dataset 5: Remove all columns with >70% NaNs and replace the rest with zero
    x_tr5, x_te5 = change_features(x_tr, x_te, threshold=0.7, replace_with_zero=True)

    # print("Pre process: initial {}, after cleaning {}".format(x_tr.shape, x_tr2.shape))
    # x_tr = x_tr5
    # x_te = x_te5
    # Dataset 6: Dataset 5 plus build_poly 7
    x_tr5_norm, x_te5_norm = standardize_features_before(x_tr5, x_te5)
    x_tr6 = build_poly(x_tr5_norm, 9)
    x_te6 = build_poly(x_te5_norm, 9)

    x_tr7, x_te7 = combine(x_tr6, x_te6)

    # build_poly and combinations
    x_tr8 = np.c_[x_tr6, x_tr7]
    x_te8 = np.c_[x_te6, x_te7]

    # Dataset 9: Remove columns 20, 18, 15
>>>>>>> 345df4de1bf1cd0c2c14172b766a8d259a770b03
    index = [20, 18, 15]
    x_te = np.delete(x_te, index, axis=1)
    x_tr = np.delete(x_tr, index, axis=1)

    x_tr1, x_te1 = combine(x_tr, x_te)
    x_tr1[np.isnan(x_tr1)] = 0
    x_te1[np.isnan(x_te1)] = 0
    x_tr[np.isnan(x_tr)] = 0
    x_te[np.isnan(x_te)] = 0
    x_tr, x_te = standardize_features_before(x_tr, x_te)
    x_tr = np.c_[build_poly(x_tr, 9), x_tr1]
    x_te = np.c_[build_poly(x_te, 9), x_te1]
    # print(x_tr[:1])
    return x_tr1, y_tr, x_te1, y_te, ids_te


def cross_validation(y, x, k_indices, model_function):
    """return the loss of ridge regression."""

    losses_tr = []
    losses_te = []
    accuracies = []

    for i, ind_te in enumerate(k_indices):
        y_te = y[ind_te]
        x_te = x[ind_te]

        ind_tr = np.vstack((k_indices[:i], k_indices[i + 1 :])).flatten()
        y_tr = y[ind_tr]
        x_tr = x[ind_tr]

        weights, loss = model_function(y_tr, x_tr)

        losses_tr.append(compute_rmse(y_tr, x_tr, weights))
        losses_te.append(compute_rmse(y_te, x_te, weights))

        # calculated how many data-points are correctly predicted
        # This is different from the loss of the text set, since at the end we're
        # only interested on the prediction
        predictions_y = predict_labels(weights, x_te)
        diff = predictions_y - y_te
        accuracy = (len(diff) - np.count_nonzero(diff)) / len(diff)
        accuracies.append(accuracy)

    avg_loss_tr = np.average(losses_tr)
    avg_loss_te = np.average(losses_te)
    avg_accuracy = np.average(accuracies)

    return avg_loss_tr, avg_loss_te, avg_accuracy


class Model:
    def __init__(self, name):
        self.name = name

    def run(self, y_training, x_training, k_indices, model_function):
        print("Run", self.name)
        (loss_training, loss_test, accuracy) = cross_validation(
            y_training, x_training, k_indices, model_function
        )
        self.loss_training = loss_training
        self.loss_test = loss_test
        self.accuracy = accuracy

        # run it with the whole data set
        (weight, total_loss) = model_function(y_training, x_training)
        self.weights = weight

    def print(self):
        print(
            np.round(self.loss_training, 4),
            np.round(self.loss_test, 4),
            np.round(self.accuracy, 4),
            self.name,
        )


def try_different_models(x_training, y_training):
    print("Try different models")

    # Let's run this in a pool, so we can use all the available CPU Cores
    # 8 ist just a number which I've chosen ( no deeper meaning )
    with Pool(8) as pool:
        arguments = []
        arguments.append((x_training, y_training))

        # submit jobs to try all degrees
        best_model_for_degree = pool.map(try_all_models_for_degree, arguments)

        best_overall_model = min(
            best_model_for_degree, key=lambda model: model.accuracy
        )
        return best_overall_model


def try_all_models_for_degree(degree_and_data):
    x_training, y_training = degree_and_data
    print("Start Degree = ")

    lambda_ = 0.0001
    max_iters = 100 ** 2
    gamma = 0.0005

    k_indices = build_k_indices(y_tr, 5)

    # m_least_square = Model("Least Square")
    # m_least_square.run(y_training, x_training, k_indices, least_squares)

    # model_function = lambda y, x: least_squares_GD(
    #     y, x, np.zeros(x.shape[1]), max_iters, gamma
    # )
    # m_least_square_gd = Model("Least Square GD")
    # m_least_square_gd.run(y_training, x_training, k_indices, model_function)

    # model_function = lambda y, x: least_squares_SGD(
    #     y, x, np.zeros(x.shape[1]), max_iters, gamma
    # )
    # m_least_square_sgd = Model("Least Square SGD")
    # m_least_square_sgd.run(y_training, x_training, k_indices, model_function)

    # model_function = lambda y, x: logistic_regression(
    #     y, x, np.zeros(x.shape[1]), max_iters, gamma
    # )
    # m_logistic_regression = Model("Logistic Regression")
    # m_logistic_regression.run(y_training, x_training, k_indices, model_function)

    # model_function = lambda y, x: reg_logistic_regression(
    #     y, x, lambda_, np.zeros(x.shape[1]), max_iters, gamma
    # )
    # m_reg_logistic_regression = Model("Reg. Logistic Regression")
    # m_reg_logistic_regression.run(y_training, x_training, k_indices, model_function)

    model_function = lambda y, x: ridge_regression(y, x, lambda_)
    m_ridge_regression = Model("Ridge Regression")
    m_ridge_regression.run(y_training, x_training, k_indices, model_function)

    # all_models = [
    #     m_least_square,
    #     m_logistic_regression,
    #     m_least_square_gd,
    #     m_least_square_sgd,
    #     m_reg_logistic_regression,
    #     m_ridge_regression,
    # ]

    all_models = [m_ridge_regression]

    # After all are done, print result
    for each_model in all_models:
        each_model.print()

    # get best model based on the loss
    best_model = min(all_models, key=lambda each: each.accuracy)
    return best_model


np.random.seed(10)
x_tr, y_tr, x_te, y_te, ids_te = read_train_test()
x_tr, x_te = standardize_features(x_tr, x_te)
best_overall_model = try_different_models(x_tr, y_tr)
print("Best model is:")
best_overall_model.print()

print(best_overall_model.weights)
predict_and_generate_file(best_overall_model.weights, x_te, ids_te)
