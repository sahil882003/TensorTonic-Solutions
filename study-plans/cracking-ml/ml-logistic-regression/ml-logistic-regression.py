import numpy as np

def _sigmoid(x):

    return 1/(1 + np.exp(-x))

def logistic_regression(X, y, lr=0.01, n_iters=1000):

    X = np.array(X,dtype = float)
    y = np.array(y,dtype = float)

    X = np.pad(X,((0,0),(0,1)),mode = 'constant', constant_values = 1)

    w = np.zeros(X.shape[1],dtype = float)
    xt = X.T

    for i in range(n_iters):

        w =  w - lr * (1/X.shape[0]) * (xt @ (_sigmoid(X@w) - y))

    return [w[:-1],w[-1]]

    
