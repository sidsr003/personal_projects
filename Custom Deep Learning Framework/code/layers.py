import numpy as np
import initializers
import optimizers

class Layer():
    """ Template class for all layers
    """

    def __init__(self):
        pass
    def initialize_parameters(self, shape, initializer):
        raise NotImplementedError
    def forward_propagate(self, X):
        raise NotImplementedError
    def backward_propagate(self, dA):
        raise NotImplementedError
    def update_parameters(self):
        raise NotImplementedError

class Dense(Layer):
    """ 
    Densely connected layer of neurons\n
    """

    def __init__(self, input_size, output_size, activation=None, kernel_initializer=initializers.Uniform(-0.5, 0.5),
                 bias_initializer = initializers.Constant(), weight_optimizer=optimizers.Vanilla_Gradient_Descent(),
                 bias_optimizer=optimizers.Vanilla_Gradient_Descent()):
        """
        Parameters\n
        ---------------\n
        input_size: 'int', the input size of the layer\n
        output_size: 'int', the output size of the layer\n
        actvation: subclass of activations.Activation, not an instance of it\n
        kernel_initializer: instance of subclass of initializers.Initializer\n
        bias_initializer: instance of subclass of initializers.Initializer\n
        optimizer: instance of subclass of optimizers.Optimizer
        """
        super().__init__()
        self.input_size = input_size # input size of the layer
        self.output_size = output_size # output size of the layer
        self.activation = activation # activation function for the layer
        self.optimizers = {'weight_optimizer': weight_optimizer, 'bias_optimizer' : bias_optimizer}
        self.parameters = {} # stores trainable layer parameters
        self.parameters['W'] = self.initialize_parameters((output_size, input_size), kernel_initializer) # (output_size, input_size) array of weights
        self.parameters['b'] = self.initialize_parameters((output_size, 1), bias_initializer) # (output_size, 1) array of biases
        self.cache = {} # stores layer computations X, Z, A for backpropagation 
        self.gradients = {} # stores gradients for parameter update

    def initialize_parameters(self, shape, initializer):
        """
        Parameters\n
        -------------\n
        shape: tuple indicating shape of initialization\n
        initializer: instance of subclass of initializers.Initializer\n
        Returns\n
        -------------\n
        ndarray of shape=shape\n
        """
        return initializer.initialize(shape)
    
    def forward_propagate(self, X):
        """
        Parameters\n
        -------------\n
        X: ndarray of shape (batch_size, input_size)\n 
        Returns\n
        -------------\n
        ndarray of shape (batch_size, output_size), the layer output\n
        """
        Z = (np.dot(self.parameters['W'], X.T) + self.parameters['b']).T # Z of shape (batch_size, output_size)
        if self.activation != None:
            A = self.activation.function(Z)
        else:
            A = Z
        self.cache['X'] = X
        self.cache['Z'] = Z
        self.cache['A'] = A
        return A
    
    def backward_propagate(self, dA):
        """
        Parameters\n
        --------------\n
        dA: ndarray of shape (batch_size, output_size), the derivative of loss with respect to output A of layer\n
        Returns\n
        --------------\n
        ndarray of shape (batch_size, input_size), derivative of loss with respect to input X\n
        """
        if self.activation != None:
            activation_derivative = self.activation.derivative(self.cache['Z'])
            dZ = np.zeros((dA.shape[0], dA.shape[1])) # dZ of shape (batch_size, output_size)
            for i in range(dA.shape[0]):
                dZ[i] = np.dot(activation_derivative[i].T, dA[i].reshape(-1, 1)).flatten()
        else:
            dZ = dA
        dW = np.dot(dZ.T, self.cache['X']) # dW of shape (output_size, input_size)
        db = np.dot(dZ.T, np.ones((dZ.shape[0], 1))) # db of shape (output_size, 1)
        dX = np.dot(dZ, self.parameters['W']) # dX of shape (batch_size, input_size)
        self.gradients['dZ'] = dZ # stores dZ in gradients dictionary
        self.gradients['dW'] = dW # stores dW in gradients dictionary 
        self.gradients['db'] = db # stores db in gradiennts dictionary
        return dX

    def update_parameters(self):
        """ 
        """
        self.parameters['W'] = self.optimizers['weight_optimizer'].optimize(self.parameters['W'], self.gradients['dW'])
        self.parameters['b'] = self.optimizers['bias_optimizer'].optimize(self.parameters['b'], self.gradients['db'])
