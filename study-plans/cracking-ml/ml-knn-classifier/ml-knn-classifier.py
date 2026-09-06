import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """
    Returns: A list of predicted integer labels for each test point
    """

    X_train = np.array(X_train,dtype = float)
    y_train = np.array(y_train,dtype = float)
    X_test = np.array(X_test,dtype = float)
    X_test = X_test[::,None]

    diffs = (X_train - X_test)
    distances = (np.sum(diffs**2,axis =2))

    idx = np.argpartition(distances, k-1, axis=1)[:,:k]

    votes = (y_train[idx])


    predictions = np.array([
    np.bincount(v.astype(int)).argmax()
    for v in votes
    ])
    
        
    return predictions
