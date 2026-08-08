import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""

    data = np.array(data,dtype = float)

    ele_0 = (data > threshold).astype(float)
    ele_1 = np.any(data > threshold,axis = 1).astype(float)
    ele_2 = np.all(data > threshold, axis = 1).astype(float)
    column_mask1 = ele_1.reshape(ele_1.shape[0],1)
    column_mask2 = ele_2.reshape(ele_2.shape[0],1)

    return np.array([ele_0,data * column_mask1,data * column_mask2])

    