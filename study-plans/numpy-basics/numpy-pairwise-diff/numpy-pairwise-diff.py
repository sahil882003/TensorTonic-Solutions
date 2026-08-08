import numpy as np

def pairwise_diff(a):
    """Returns: np.ndarray of shape (n, n) where out[i,j] = a[i] - a[j]"""
    data = np.array(a,dtype = float)
    return data.reshape(-1,1) - data