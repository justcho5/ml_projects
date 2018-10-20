#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 19:50:45 2018

@author: natirodri
"""
from implementations import (
    least_squares_GD,
    least_squares_SGD,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    build_poly
)

from helpers import load_csv_data, predict_labels, create_csv_submission

from cross_val import cross_validation, build_k_indices
from multiprocessing import Pool

import numpy as np


def remove_features(x_tr, x_te, threshold=0, replace_with_mean=False):
    #     print("First five train rows before:\n", x_tr[:5], "\n")
    #     print("First five test rows before:\n", x_te[:5], "\n")

    mask = np.isnan(x_tr).sum(axis=0) / x_tr.shape[0] > threshold
    x_te = x_te[:, ~mask]
    x_tr = x_tr[:, ~mask]
    if replace_with_mean:
        col_mean = np.nanmean(x_tr, axis=0)
        #         print("COLUMN MEAN:\n", col_mean, "\n")
        nan_inds_tr = np.where(np.isnan(x_tr))
        nan_inds_te = np.where(np.isnan(x_te))

        x_te[nan_inds_te] = np.take(col_mean, nan_inds_te[1])
        x_tr[nan_inds_tr] = np.take(col_mean, nan_inds_tr[1])
    #     print("First five train rows after:\n", x_tr[:5], "\n")
    #     print("First five test rows after:\n", x_te[:5], "\n")
    #     print("Number of NaNs per training feature:\n", np.isnan(x_tr).sum(axis=0), "\n")
    #     print("Number of NaNs per test feature:\n", np.isnan(x_te).sum(axis=0), "\n")

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
