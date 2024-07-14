import numpy as np
import math

class Loss():
    def __init__(self):
        pass
    def __call__(self, y_predict, y):
        return self.function(y_predict, y)
    @staticmethod
    def function(y_predict, y):
        pass
    @staticmethod
    def derivative(y_predict, y):
        pass

class MSE(Loss):
    def _init__(self):
        super().__init__()
    @staticmethod
    def function(y_predict, y):
        return np.sum(np.square(y_predict-y))/(y_predict.shape[0]*y_predict.shape[1]) # MSE loss
    @staticmethod
    def derivative(y_predict, y):
        return 2*(y_predict-y)/(y_predict.shape[0]*y_predict.shape[1]) # MSE loss


class Cross_Entropy(Loss):
    def __init__(self):
        super().__init__()
    @staticmethod
    def function(y_predict, y):
        epsilon = 1e-7
        return -np.sum(y*(np.log(np.maximum(y_predict, epsilon))))/y_predict.shape[0]
    @staticmethod
    def derivative(y_predict, y):
        epsilon = 1e-7
        return -(y/np.maximum(y_predict, epsilon))/y_predict.shape[0]

  
