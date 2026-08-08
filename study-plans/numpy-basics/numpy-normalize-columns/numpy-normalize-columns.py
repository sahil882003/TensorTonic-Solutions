import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""

    data = np.array(data, dtype = float)

    return (data - np.mean(data, axis = 0))/np.std(data,axis = 0)