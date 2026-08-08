import numpy as np

def outer_sum(a, b):
    """Returns: np.ndarray of shape (m, n), outer sum where out[i,j] = a[i] + b[j]"""
    a = np.array(a,dtype = float)
    b = np.array(b,dtype = float)
    return a.reshape(-1,1) + b