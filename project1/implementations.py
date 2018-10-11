# Return type: Note that all functions should return: (w, loss), which is the last weight vector of the
# method, and the corresponding loss value (cost function). Note that while in previous labs you might have
# kept track of all encountered w for iterative methods, here we only want the last one.


def least_squares_GD(y, tx, initial w, max iters, gamma):
# Linear regression using gradient descent
def least_squares_SGD(y, tx, initial w, max iters, gamma):
# Linear regression using stochastic gradient descent
def least_squares(y, tx):
# Least squares regression using normal equations
def ridge_regression(y, tx, lambda_):
# Ridge regression using normal equations
def logistic_regression(y, tx, initial w, max iters, gamma):
# Logistic regression using gradient descent or SGD
def reg_logistic_regression(y, tx, lambda_, initial w, max iters, gamma):


