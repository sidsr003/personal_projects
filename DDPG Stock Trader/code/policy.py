import numpy as np
from noise import Ornstein_Uhlenbeck

class Ornstein_Uhlenbeck_Noisy_Policy():
    def __init__(self):
        self.noise_generator = Ornstein_Uhlenbeck(0.1, 0.2, 0.01)
        self.epsilon = 0
    def sample_policy(self, actor, state):
        action = actor(state).numpy()
        self.epsilon = self.noise_generator.sample(self.epsilon)
        action = action + np.random.normal(0.05)
        return action

# import matplotlib.pyplot as plt
# noise = Ornstein_Uhlenbeck(1, 0.2, 0.01)
# x = 0
# xs=[]
# for i in range(1000):
#     x = noise.sample(x)
#     xs.append(x)
# plt.plot(xs)
# plt.show()