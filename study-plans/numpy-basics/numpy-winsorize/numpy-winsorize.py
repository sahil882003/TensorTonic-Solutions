import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    
    data = np.array(data,dtype = float)
    low_bnd = np.percentile(data,axis = 0, q = lo_q)
    up_bound = np.percentile(data,axis = 0,q = hi_q)
    clipped_data = np.clip(data,low_bnd,up_bound)
    masked_low = (data < low_bnd).astype(float)
    masked_up = (data > up_bound).astype(float)
    return np.stack([clipped_data,masked_low,masked_up])
    