import numpy as np
from layers import Layer

class ActivationLayer(Layer):
    def __init__(self, activation, activationDerivative):
        self.activation = activation
        self.activationDerivative = activationDerivative

    def forwardPropagation(self, inputData):
        self.input = inputData
        self.output = self.activation(self.input)
        return self.output
    
    def backwardPropagation(self, outputError, learningRate): #Learning rate not required for activation layers
        inputError = outputError*self.activationDerivative(self.input) #This is corresponding position multiplication for numpy arrays of any matching dimensions
        return inputError
        
    
