import numpy as np

def MSE(y_true, y):
    return np.mean(np.power(y_true-y, 2)) #We calculate the mean of all the elements in the column vector of errors for that sample
def MSEDerivative(y_true, y):
    return( (2*(y-y_true))/y_true.shape[0])
