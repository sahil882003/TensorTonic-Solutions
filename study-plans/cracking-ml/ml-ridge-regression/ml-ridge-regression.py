def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """

    X = np.array(X,dtype = float)
    y = np.array(y,dtype = float)
    n,d = X.shape
    W = np.zeros(d)
    b = 0
    xtx = X.T@X
    xt = X.T
    for i in range(epochs):

        prediction_actual_diff = ((X @ W + b) - y)
        delta_w = (2/n * X.T @ (prediction_actual_diff)) + (2 * alpha * W)
        delta_b = 2/n * np.sum(prediction_actual_diff)
        W = W - lr * delta_w
        b = b - lr * delta_b

    return [W,b]
        
        