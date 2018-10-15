def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


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
