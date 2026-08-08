import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.default_rng(seed)
    
    if kind == "uniform":
        return rng.uniform(size = shape)
    else:
        return rng.normal(size = shape)
