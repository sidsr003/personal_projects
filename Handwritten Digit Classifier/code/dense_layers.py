import numpy as np
from layers import Layer

class DenseLayer(Layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.rand(outputSize, inputSize) - 0.5
        self.biases = np.random.rand(outputSize, 1) - 0.5

    def forwardPropagation(self, inputData):
        self.input = inputData
        self.output = np.dot(self.weights, self.input) + self.biases
        return self.output
    
    def backwardPropagation(self, outputError, learningRate): #Note that the outputError, weightsError, biasesError are actually the value of the partial derivative of E wrt y, W and b respectively
        inputError = np.dot(self.weights.T, outputError)
        weightsError = np.dot(outputError, self.input.T)
        biasesError = outputError

        #Time to update the weights and biases for the neuron layer
        self.weights = self.weights - weightsError*learningRate
        self.biases = self.biases - biasesError*learningRate
        return inputError
        
                    
