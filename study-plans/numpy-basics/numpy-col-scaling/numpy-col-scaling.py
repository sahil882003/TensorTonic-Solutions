import numpy as np

def scale_cols(data, weights):
    """Returns: np.ndarray of shape (m, n), each column scaled by corresponding weight"""
    return np.array(data,dtype = float) * np.array(weights, dtype = float)