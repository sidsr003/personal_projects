import numpy as np

class Initializer():
    """Note that some class initializers are meant for weight initialization only, and not biases

    Several initialization techniques are specific to 2D weight matrices with a non-trivial input_shape ~and~ output shape
    """

    def __init__(self):
        pass
    @staticmethod
    def initialize():
        raise NotImplementedError

class Constant(Initializer):
    """Constant value initialization
    """

    def __init__(self, constant_value=0):
        """Parameters\n
        ---------------\n
        constant_value: the constant number to initialize parameters as\n
        """
        self.constant_value = constant_value
    def initialize(self, shape):
        """Parameters\n
        ---------------
        shape: tuple containing required dimensions along each axis
        Returns\n
        ---------------\n
        ndarray of shape=shape\n
        """
        return np.zeros(shape=shape) + self.constant_value

class Uniform(Initializer):
    """Sampled from a uniform distribution\n
    """

    def __init__(self, a, b):
        """Parameters\n
        ---------------\n
        a, b: the minimum and maximum range to generate uniformly distributed random values within\n
        """
        super().__init__()
        self.a = a
        self.b = b
    def initialize(self, shape):
        """Parameters\n
        ---------------\n
        shape: tuple containing required dimensions along each axis\n
        Returns\n
        ---------------\n
        ndarray of shape=shape\n
        """
        return np.random.uniform(self.a, self.b, size=shape)

class Xavier(Initializer):
    """ Meant for weight initialization ONLY\n
        Sampled from a standard normal distribution with standard deviation = 1/sqrt(input_size)\n
    """

    def __init__(self):
        super().__init__()
    def initialize(self, shape):
        """Parameters\n
        ---------------
        shape: tuple containing required dimensions along each axis\n
        Returns\n
        ---------------\n
        ndarray of shape=shape\n
        """
        return np.random.normal(loc=0, scale=1/np.sqrt(shape[1]), size=shape) # shape[1] = input_size

    