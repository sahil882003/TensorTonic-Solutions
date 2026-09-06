import numpy as np


def array_exp(x):
    return np.exp(x)


def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X,dtype = float)
    y = np.array(y,dtype = float)
    n,d = X.shape

    weights = np.zeros((d,n_classes))
    biases = np.zeros(n_classes)

    
    

    for i in range(n_iters):
        raw_scores = X @ weights + np.ones(n).reshape(-1,1) * biases
        raw_scores = raw_scores - np.max(raw_scores,axis = 1).reshape(-1,1)
        raw_scores = array_exp(raw_scores)
        class_probs = raw_scores / np.sum(raw_scores, axis = 1).reshape(-1,1)
        for j in range(n_classes):
            curr_cls_enc = (y == j).astype(float)
            errors = class_probs[:,j] - curr_cls_enc
            errors = errors.reshape(-1,1)
            weights[:,j] = weights[:,j] - (lr * 1/n * (X.T @ (errors)).reshape(d))
            biases[j] = biases[j] - (lr * 1/n * (np.sum(errors)))


    return weights,biases
        

        

    
        
