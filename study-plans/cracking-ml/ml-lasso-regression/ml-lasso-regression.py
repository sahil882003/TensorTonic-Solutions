def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """


    X = np.array(X,dtype = float)
    y = np.array(y,dtype = float)
    n,d = X.shape

    w = np.zeros(d)
    b = 0

    for i in range(epochs):

        errors = X @ w + b - y
        dw = 2/n * (X.T @ errors) + alpha * np.sign(w)
        db = 2/n * np.sum(errors)

        w = w - lr * (dw)
        b = b - lr * (db)


    return [w,b]
    