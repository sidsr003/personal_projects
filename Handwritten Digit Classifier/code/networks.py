import numpy as np
class Network():
    def __init__(self):
        self.layers = []
        self.loss = None
        self.lossDerivative = None
        
    def addLayer(self, layerToAdd):
        self.layers.append(layerToAdd)

    def useLoss(self, lossToUse, lossDerivativeToUse):
        self.loss = lossToUse
        self.lossDerivative = lossDerivativeToUse

    def predict(self, inputData): #Input data will be a set of input vectors (sampleCount x inputSize x 1)
        sampleCount = inputData.shape[0]
        result = []

        for i in range(sampleCount):
            output = inputData[i]
            for layer in self.layers:
                output = layer.forwardPropagation(output)
                
            result.append(output)
            
        return result

    def fit(self, x_train, y_train, epochs, learningRate): #Epochs are the number of passes through the whole dataset
        for i in range(epochs):
            displayError = 0
            for j in range(x_train.shape[0]):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forwardPropagation(output)
                displayError +=  self.loss(y_train[j], output)
                
                error = self.lossDerivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backwardPropagation(error, learningRate)

            displayError /= x_train.shape[0]
            print("Epoch {0}/{1} complete. Error = {2}".format(i+1, epochs, displayError))         
