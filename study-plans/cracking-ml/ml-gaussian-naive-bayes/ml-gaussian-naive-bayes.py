import numpy as np
from scipy.stats import norm

def fit_guassian(data):

    mean_ = np.mean(data,axis = 0)
    std_ = np.std(data,axis = 0)
    std_ += 10**(-9)

    return np.array([mean_,std_])
    
    
    
def gaussian_nb(X_train, y_train, X_test):
    """
    Returns: A list of predicted integer labels for each test point
    """
    x_train = np.array(X_train, dtype = float)
    y_train = np.array(y_train, dtype = int)

    unq = np.unique(y_train)
    arr = []
    priors = []
    for i in unq:
        class_data = x_train[y_train == i]
        arr.append(fit_guassian(class_data))
        priors.append(((y_train == i).astype(float).sum())/y_train.shape[0])
        
        

    params = np.array(arr)
    ans = []
    for test_data in X_test:
        max = -1
        max_value = -np.inf
        for class_ in unq:
            probs = norm.logpdf(test_data,loc = params[class_][0],scale = params[class_][1])
            probs = np.array(probs,dtype = float)

            total_prob = np.log(priors[class_]) + np.sum(probs)
            if total_prob > max_value:
                max_value = total_prob
                max = class_
        ans.append(max)

    return ans

    
    