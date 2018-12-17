# -*- coding: utf-8 -*-

from scipy.optimize import minimize

def sum_ratings(weights):
    mix_prediction = 0
    for i, pred in enumerate(all_predictions_test):
        mix_prediction += weights[i] * pred['raw_ratings'].values

    mix_prediction = mix_prediction.clip(1, 5)

    return sqrt(mean_squared_error(mix_prediction,
                                   make_df_from_list(testset)['raw_ratings'].values))

w0 = [1 / len(all_predictions_test)] * len(all_predictions_test)
result = minimize(fun=sum_ratings, x0=w0)


mix_prediction = 0
for i, pred in enumerate(all_predictions_test):
    mix_prediction += result[i] * pred['raw_ratings'].values

mix_prediction = mix_prediction.clip(1, 5)

mix_prediction = 0
for i, pred in enumerate(all_predictions_test):
    mix_prediction += clf.coef_[i] * pred['raw_ratings'].values

mix_prediction += clf.intercept_
mix_prediction = mix_prediction.clip(1, 5)
