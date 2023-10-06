import numpy as np

def sigmoid(inputData):
    return 1/(1+np.exp(-1*inputData))
def sigmoidDerivative(inputData):
    numerator = np.exp(-1*inputData)
    denominator = np.power(1+np.exp(-1*inputData), 2)
    return(numerator/denominator)
def tanh(inputData):
    return np.tanh(inputData)
def tanhDerivative(inputData):
    return 1-np.power(np.tanh(inputData), 2)
