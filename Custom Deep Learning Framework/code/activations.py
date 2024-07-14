import numpy as np
import math

class Activation():
    def __init__(self):
        pass
    def __call__(self, x):
        return self.function(x)
    @staticmethod
    def function(x):
        raise NotImplementedError
    @staticmethod
    def derivative(x):
        raise NotImplementedError

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
    @staticmethod
    def function(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def derivative(x):
        # returns vector of jacobians of shape (m, (jacobian shape)) = (m, output_shape, output_shape)
        derivative = np.exp(-x)/(1+np.exp(-x))**2
        # convert to required form (m, (jacobian shape)) = (m, output_shape, output_shape)
        reshaped_derivative = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            reshaped_derivative[i] = np.diag(derivative[i].flatten())
        return reshaped_derivative
    
class Softmax(Activation):
    def __init__(self):
        super().__init__()
    @staticmethod
    def function(x):
        # print(x)
        x = x-np.max(x, axis=1).reshape(-1, 1)
        f = np.exp(x)/(np.sum(np.exp(x), axis=1).reshape(-1, 1))
        return f
    @staticmethod
    def derivative(x):
        x = x-np.max(x, axis=1).reshape(-1, 1)
        f = np.exp(x)/(np.sum(np.exp(x), axis=1).reshape(-1, 1))

        derivative = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            derivative[i] = np.diag(f[i].flatten()) - np.dot(f[i].reshape(-1, 1), f[i].reshape(-1, 1).T)
        return derivative
    
class Relu(Activation):
    def __init__(self):
        super().__init__()
    @staticmethod
    def function(x):
        return np.where(x>0, x, 0)
    @staticmethod
    def derivative(x):
        # returns vector of jacobians of shape (m, (jacobian shape)) = (m, output_shape, output_shape)
        derivative = np.where(x>0, 1, 0)

        # convert to required form (m, (jacobian shape)) = (m, output_shape, output_shape)
        reshaped_derivative = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            reshaped_derivative[i] = np.diag(derivative[i].flatten())
        return reshaped_derivative
    
class Tanh(Activation):
    def __init__(self):
        super().__init__()
    @staticmethod
    def function(x):
        return np.tanh(x)
    @staticmethod
    def derivative(x):
        # returns vector of jacobians of shape (m, (jacobian shape)) = (m, output_shape, output_shape)
        derivative = (1/np.cosh(x))**2

        # convert to required form (m, (jacobian shape)) = (m, output_shape, output_shape)
        reshaped_derivative = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            reshaped_derivative[i] = np.diag(derivative[i].flatten())
        return reshaped_derivative
