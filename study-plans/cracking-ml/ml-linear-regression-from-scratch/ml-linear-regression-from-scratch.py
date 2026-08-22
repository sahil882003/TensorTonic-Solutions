import numpy as np

def linear_regression(X, y, lr, epochs):
    """ 
    Returns: tuple (weights, bias)
    """

    X = np.array(X,dtype = float)
    X = np.pad(X,((0,0),(0,1)),mode = 'constant',constant_values = 1)
    total_weights = len(X[0])
    total_datapoints = len(X)
    weights = np.zeros(total_weights).T

    xtx = X.T @ X
    xt = X.T


    for i in range(epochs):
        weights = weights - (lr * (2/total_datapoints) * (xtx @ weights - xt @ y))

    return [weights[:-1],weights[-1]]
