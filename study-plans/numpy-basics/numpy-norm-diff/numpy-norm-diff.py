import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""

    a = np.array(a,dtype = float)
    b = np.array(b,dtype = float)
    a = np.clip(a,lo,hi)
    b = np.clip(b,lo,hi)

    a = a/(hi -lo)
    b = b/(hi - lo)
    return np.abs(a - b)
