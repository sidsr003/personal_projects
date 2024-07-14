import numpy as np

class Buffer():
    """ 
    The buffer class stores a finite number of the latest historically observed observations in the time-series as an array of MDP tuples (state, action, next_state, reward) 
    """

    def __init__(self, buffer_size, batch_size, state_size, action_size):
        """
        Parameters\n
        ------------------\n
        buffer_size: the size of memory buffer used to store experience\n
        batch_size: the size of sample batches to be drawn using experience replay\n
        state_space_shape: tuple describing the dimensions of the state space of the MDP\n
        action_space_shape: tuple describing the dimesnions of the action space of the MDP\n 
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_buffer = None # Stores the states from each experience
        self.action_buffer = None # Stores the actions from each experience
        self.next_state_buffer = None # Stores the next states from each experience
        self.reward_buffer = None # Stores the rewards from each experience
        self.historical_occupancy = 0 # Stores the total number of observations ever recorded
        self.active_index = 0 # Stores the current active index to write new data to
        self.initialize_buffer(state_size, action_size) # 

    def initialize_buffer(self, state_size, action_size):
        """
        Initializes MDP buffers to zero-arrays\n
        Parameters\n
        ---------------
        state_space_shape: tuple describing the dimensions of the state space of the MDP\n
        action_space_shape: tuple describing the dimesnions of the action space of the MDP\n
        Returns\n
        ---------------
        """
        self.state_buffer = np.zeros(shape=(self.buffer_size, state_size)) # shape=(buffer_size, <state_space_shape>) 
        self.action_buffer = np.zeros(shape=(self.buffer_size, action_size)) # shape=(buffer_size, <action_space_shape>)
        self.next_state_buffer = np.zeros(shape=(self.buffer_size, state_size)) # shape=(buffer_size, <state_space_shape>)
        self.reward_buffer = np.zeros(shape=(self.buffer_size, 1))
    
    def write_to_buffer(self, observation):
        """Implement buffer writing of a single observation (state, action, next_state, reward) of the environment-actor interaction"""
        
        self.state_buffer[self.active_index] = observation[0]
        self.action_buffer[self.active_index] = observation[1]
        self.next_state_buffer[self.active_index] = observation[2]
        self.reward_buffer[self.active_index, 0] = observation[3]
        self.active_index = (self.active_index + 1) % self.buffer_size
        self.historical_occupancy += 1
 
    def sample_buffer(self):
        # Implement buffer sampling of a random sample of observations to perform monte carlo estimation and subsequent learning
        sampling_indices = np.random.choice(range(min(self.historical_occupancy, self.buffer_size)),  min(self.batch_size, self.historical_occupancy))
        sample_observations = (self.state_buffer[sampling_indices], self.action_buffer[sampling_indices], self.next_state_buffer[sampling_indices], self.reward_buffer[sampling_indices])
        return sample_observations
    
