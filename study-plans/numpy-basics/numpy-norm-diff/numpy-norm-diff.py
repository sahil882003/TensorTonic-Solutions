import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a = np.array(a,dtype = float)
    a = np.clip(a, lo,hi)
    a = a/(hi - lo)
    b = np.array(b,dtype = float)
    b = np.clip(b,lo,hi)
    b = b/(hi - lo)
    return np.abs(a- b)