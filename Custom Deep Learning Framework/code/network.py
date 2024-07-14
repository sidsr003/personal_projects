from metrics import accuracy
from copy import deepcopy

class Sequential():
    def __init__(self, layers):
        # expects layers: list of layers
        self.layers = layers
    def compile(self, loss, optimizer=None, metrics=['accuracy']):
        self.loss = loss
        self.metrics = metrics
        if optimizer!=None:
            for layer in self.layers:
                for key in layer.optimizers:
                    layer.optimizers[key] = deepcopy(optimizer)
    def summary(self):
        for layer in self.layers:
            print(f"Layer type: {layer.__class__.__name__}, input_size:{layer.input_size}, output_size:{layer.output_size}")
    def fit(self, X, y, batch_size=1, epochs=1):
        # expects X of shape (num_examples, input_size)
        # expects y of shape (num_examples, output_size)
        X = X.reshape(X.shape[0]//batch_size, batch_size, X.shape[1])
        y = y.reshape(y.shape[0]//batch_size, batch_size, y.shape[1])
        for i in range(epochs):
            loss = 0
            for j in range(X.shape[0]):
                y_pred = X[j]
                for layer in self.layers:
                    y_pred = layer.forward_propagate(y_pred)
                loss += self.loss.function(y_pred, y[j])
                loss_derivative = self.loss.derivative(y_pred, y[j]) # loss with respect to y_pred
                dX = loss_derivative
                for layer in reversed(self.layers):
                    dX = layer.backward_propagate(dX)
                for layer in self.layers:
                    layer.update_parameters()
            loss /= X.shape[0]
            print(f"Completed epoch {i+1} with epoch average loss = {loss}")

    def predict(self, X):
        # expects X of shape (num_examples, input_size)
        for layer in self.layers:
            X = layer.forward_propagate(X)
        return X
    def validate(self, X, y):
        # expects X of shape (num_examples, input_size)
        # expects y of shape (num_examples, output_size)
        scores = {}
        for layer in self.layers:
            X = layer.forward_propagate(X)
        for metric in self.metrics:
            if metric == 'accuracy':
                score = accuracy(X, y)
                scores['accuracy'] = score
        return scores 
            

