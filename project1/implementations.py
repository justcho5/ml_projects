# Return type: Note that all functions should return: (w, loss), which is the last weight vector of the
# method, and the corresponding loss value (cost function). Note that while in previous labs you might have
# kept track of all encountered w for iterative methods, here we only want the last one.

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


def compute_mse(y,tx,w):
    e = y-np.dot(tx,w)
    mse = (1/2)*np.mean(e**2)
    return mse
def compute_gradient(y, tx, w):
    N = y.shape[0]
    e = np.subtract(y, np.dot(tx,w))
    gradient =  -(1/(2*N))*np.dot((tx.T),(e))
    return gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
# Linear regression using gradient descent
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y,tx,w)
        loss = compute_mse(y,tx,w)
        w = w-gamma*(gradient)
        # store w and loss
        
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
# Linear regression using stochastic gradient descent
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
        
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = (compute_mse(minibatch_y, minibatch_tx,w))
        
            w = w-gamma*(gradient)
        # store w and loss
        
    return (w, loss)

def least_squares(y, tx):
# Least squares regression using normal equations
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a,b)
    # Here I used RMSE
    loss = np.sqrt(2*compute_mse(y, tx, w))
    return (w, loss)
def ridge_regression(y, tx, lambda_):
# Ridge regression using normal equations
    aI = 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(a+aI, b)
    # RMSE?
    loss = np.sqrt(2*compute_mse(y,tx,w))
    return (w, loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
# Logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):


