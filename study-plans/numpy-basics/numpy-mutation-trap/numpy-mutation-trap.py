import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    data = np.array(data, dtype = float)
    slice = data[row_idx , :]
    return np.array([slice, np.clip(slice,lo,hi)])