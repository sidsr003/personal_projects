import numpy as np
import matplotlib.pyplot as plt

class Noise():
    """ 
    Class to implement a noise sampling generator
    """
    def __init__(self):
        """
        Initialize Noise object
        """
        raise NotImplementedError
    def sample(self, signal):
        """
        Return generated noise sample
        """
        raise NotImplementedError

class Ornstein_Uhlenbeck(Noise):
    """
    Generates noise according to the discrete Ornstein-Uhlenbeck Process: https://math.stackexchange.com/questions/345773/how-the-ornstein-uhlenbeck-process-can-be-considered-as-the-continuous-time-anal
    """

    def __init__(self, theta, sigma, dt=0.01):
        """
        Parameters \n
        ----------------\n
        theta: OU Parameter\n
        sigma: OU Parameter\n
        dt: the step size\n

        Returns\n
        -----------------\n
        None
        """
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

    def sample(self, signal):
        """
        Parameters\n
        ----------------\n
        signal: the signal value to which autocorrelated correlated noise must be added\n
        Returns\n
        -----------------\n
        signal_propagated: the noisy signal
        """
        signal_propagated = signal*(1-self.theta*self.dt) + self.sigma*np.sqrt(self.dt)*np.random.normal()
        return signal_propagated