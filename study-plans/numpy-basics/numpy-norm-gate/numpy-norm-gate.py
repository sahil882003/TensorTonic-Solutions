import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    x = np.array(X,dtype = float)
    w = np.array(W,dtype = float)
    matrix_mul = x @ w
    l2_norm = np.linalg.norm(matrix_mul,axis = 1)
    return matrix_mul * (l2_norm >= threshold).astype(float).reshape(-1,1)