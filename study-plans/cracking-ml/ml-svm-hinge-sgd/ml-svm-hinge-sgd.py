import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    """
    Returns fitted parameters and training predictions.
    """
    data = np.array(X,dtype = np.float64)
    target = np.array(y,dtype = np.float64)

    n,d = data.shape

    weights = np.zeros(d,dtype = np.float64)
    bias = 0.0

    for i in range(n_epochs):
        
        for j in range(n):

            if 1 - target[j] * (np.dot(weights,data[j]) + bias) > 0:
                weights -= lr * (lam * weights - target[j] * data[j])
                bias -= lr * (-target[j])
            else:
                weights -= lr * (lam * weights)

    predictions = data @ weights + bias
    predictions = np.where(predictions > 0, 1,-1)
    
    ans = {
        'weights': weights.tolist(),
        'bias': bias,
        'predictions':predictions.tolist()
    }
    return ans
