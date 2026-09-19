import numpy as np

def averaged_perceptron(X: list, y: list, lr: float, epochs: int) -> dict:
    """
    Returns a dictionary: weights (float list), bias (float), predictions (integer list).
    """

    data = np.array(X,dtype = float)
    labels = np.array(y,dtype = int)
    
    n,d = data.shape

    average_weights = np.zeros(d,dtype = float)
    average_bias = 0.0
    count = 0
    
    weights = np.zeros(d,dtype = float)
    bias = 0.0

    for _ in range(epochs):
        
        for i in range(n):
    
            if (labels[i] * (np.dot(weights,data[i]) + bias)) <= 0:
                
                bias = bias + lr * labels[i]
                weights = weights + lr * labels[i] * data[i]
            average_weights += weights
            average_bias += bias
            count += 1

    average_weights = average_weights/count
    average_bias = average_bias/count 

    predictions = data @ average_weights[:None] + average_bias
    predictions = np.where(predictions > 0,1,-1)

    print(type(average_bias))
    return {"weights": [round(float(value), 4) for value in average_weights], "bias": round(float(average_bias), 4), "predictions": predictions.tolist()}
    
