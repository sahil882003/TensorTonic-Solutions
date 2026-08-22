import numpy as np

def compare_correlations(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    a = np.array(a,dtype = float)
    b = np.array(b, dtype = float)
    c = np.concat([a,b],axis = 0)
    
    corr_1 = np.corrcoef(a,rowvar = False)
    corr_2 = np.corrcoef(b,rowvar = False)
    corr_3 = np.corrcoef(c,rowvar = False)

    return np.array([corr_1,corr_2,corr_3])