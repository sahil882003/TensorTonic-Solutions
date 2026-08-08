import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    data = np.array(data,dtype = float)
    return np.array([np.sort(data,axis = axis),np.argsort(data,axis = axis).astype(float)])