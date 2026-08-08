import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    data = np.array(arr, dtype = float)
    if not axis:
        return data[indices,:]
    else:
        return data[:, indices]
        