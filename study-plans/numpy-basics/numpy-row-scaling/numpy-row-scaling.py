import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    data = np.array(data,dtype = float)
    weights = np.array(weights,dtype = float)
    return np.diag(weights) @ data