import numpy as np

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""

    pad = lambda x: np.pad(x,pad_width, mode ='constant', constant_values = 0.0)
    data = np.array(data, dtype = float)
    rounded = pad(np.round(data,decimals))
    floored = pad(np.floor(data))
    ceiled = pad(np.ceil(data))
    return np.stack([rounded, floored, ceiled])
    