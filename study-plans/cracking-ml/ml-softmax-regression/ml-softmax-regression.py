import numpy as np


def array_exp(x):
    return np.exp(x)


def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X,dtype = float)
    y = np.array(y,dtype = int)
    n,d = X.shape

    one_hot_encoded_y = np.zeros((n,n_classes))
    one_hot_encoded_y[np.arange(0,n),y] = 1

    y = one_hot_encoded_y
    
    weights = np.zeros((d,n_classes))
    biases = np.zeros(n_classes)

    
    

    for i in range(n_iters):
        raw_scores = X @ weights + np.ones(n).reshape(-1,1) * biases
        raw_scores = raw_scores - np.max(raw_scores,axis = 1).reshape(-1,1)
        raw_scores = array_exp(raw_scores)
        class_probs = raw_scores / np.sum(raw_scores, axis = 1).reshape(-1,1)
       
        gradient_weights =1/n * X.T @ (class_probs - y)
        gradient_biases = 1/n * np.sum(class_probs - y,axis = 0)

        weights = weights - lr * gradient_weights
        biases = biases - lr * gradient_biases

        

    return weights,biases
        

        

    
        
