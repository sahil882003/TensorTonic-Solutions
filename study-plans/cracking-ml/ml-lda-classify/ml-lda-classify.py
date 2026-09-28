import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    """
    Returns one predicted label for each test row.
    """
    data = np.array(X_train,dtype = float)
    target = np.array(y_train,dtype =int)
    X_test = np.array(X_test,dtype = float)
    n,d = data.shape
    labels = np.unique(y_train)
    class_count = labels.shape[0]
    mean_estimates = {}
    priors = {}
    covariance = np.zeros((d,d))
    
    
    for label in labels:
        data_points = data[target == label]
        n_temp,d_temp = data_points.shape
        current_mean = np.sum(data_points, axis = 0)*1/n_temp
        mean_estimates[label] = current_mean
        priors[label] = n_temp/n
        centered_data = data_points - current_mean
        covariance += centered_data.T @ centered_data
        
    
    covariance = covariance * 1/(n-class_count)
    covariance += 10**-6 * np.eye(d,d)
    ans = []
    for test_point in X_test:
        high_score_init = float('-inf')
        high_score_label = None
        for label in labels:
          mean_estimate = mean_estimates[label]
          prior = priors[label]
          score = np.log(prior) + test_point.T @ np.linalg.inv(covariance) @ mean_estimate - 1/2 * np.array(mean_estimate).T @ np.linalg.inv(covariance) @ mean_estimate
          if score > high_score_init:
            high_score_init = score
            high_score_label = label
        ans.append(high_score_label)
    return ans