import numpy as np

class Optimizer():
    def __init__(self):
        pass
    def optimize():
        raise NotImplementedError

class Vanilla_Gradient_Descent(Optimizer):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
    def optimize(self, parameters, gradients):
        return parameters-self.learning_rate*gradients
    
class RMS_Prop(Optimizer):
    def __init__(self, learning_rate=1e-3, decay_rate=0.9, moving_square_gradients=0, epsilon=1e-7):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.moving_square_gradients = moving_square_gradients
        self.epsilon = epsilon
    def optimize(self, parameters, gradients):
        self.moving_square_gradients = self.decay_rate*self.moving_square_gradients+(1-self.decay_rate)*gradients**2
        return parameters - self.learning_rate*gradients/(np.sqrt(self.moving_square_gradients)+self.epsilon)
        