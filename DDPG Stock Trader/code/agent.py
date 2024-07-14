import tensorflow as tf
import numpy as np
import gymnasium as gym

from actor import Actor
from critic import Critic
from policy import Ornstein_Uhlenbeck_Noisy_Policy
from buffer import Buffer
from environments import Stock_Market

class DDPG_Agent():
    """ 
    Implements the DDPG agent
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(self.state_size, self.action_size)
        self.critic = Critic(self.state_size, self.action_size)
        self.target_actor = Actor(self.state_size, self.action_size)
        self.target_critic = Critic(self.state_size, self.action_size)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.policy = Ornstein_Uhlenbeck_Noisy_Policy()
        self.buffer = Buffer(50000, 64, self.state_size, self.action_size)
        self.environment = Stock_Market(starting_balance=10000)
        self.last_state, _ = self.environment.reset()
        self.current_state = None
        self.current_action = None
        self.reward = None
        self.discount = 1 # The reward discounting factor
        self.tau = 0.005 # The momentum term for target network temporal updates
        self.balances = []
        self.balance = self.last_state[-2]
    
    def train_step(self, training, episode):
        """
        Implement training
        """
        end_of_episode = False

        self.current_action = np.squeeze(self.policy.sample_policy(self.actor, np.expand_dims(self.last_state, axis=0)), axis=0)
        current_state, self.reward, terminated, truncated, _ = self.environment.step(self.current_action)
        self.current_state = current_state

        self.balance = self.current_state[-2]
        
        if training:
            self.buffer.write_to_buffer((self.last_state, self.current_action, self.current_state, self.reward))
            sample_observations = self.buffer.sample_buffer()
            num_samples = sample_observations[0].shape[0]
            states = tf.convert_to_tensor(sample_observations[0], dtype=tf.float32)
            actions = tf.convert_to_tensor(sample_observations[1], dtype=tf.float32)
            next_states = tf.convert_to_tensor(sample_observations[2], dtype=tf.float32)
            rewards = tf.convert_to_tensor(sample_observations[3], dtype=tf.float32)
            actor_grads, critic_grads = self.compute_gradients(states, actions, next_states, rewards)
            self.update_networks(actor_grads, critic_grads)
        self.last_state = self.current_state

        self.balances.append(self.balance)
        self.print_position(episode)

        if truncated:
            self.last_state, _ = self.environment.reset()
            self.print_position(episode)
            end_of_episode = True
            
        return end_of_episode

    def compute_gradients(self, states, actions, next_states, rewards):
        with tf.GradientTape(persistent=True) as tape:
            y = rewards + self.discount*self.target_critic(next_states, self.target_actor(next_states))
            critic_loss =  tf.reduce_mean(tf.math.square(y-self.critic(states, actions)))
            J_sample = -tf.reduce_mean(self.critic(states, self.actor(states)))
        critic_grads = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        actor_grads = tape.gradient(J_sample, self.actor.model.trainable_variables)  
        return (actor_grads, critic_grads)
    

    def update_networks(self, actor_grads, critic_grads): 
        self.critic.update(critic_grads)
        self.actor.update(actor_grads)
        
        target_weights = self.target_actor.get_weights()
        source_weights = self.actor.get_weights()
        new_weights = []
        for i in range(len(target_weights)):
            new_weights.append((1-self.tau)*target_weights[i] + self.tau*source_weights[i])
        self.target_actor.set_weights(new_weights)
 
        target_weights = self.target_critic.get_weights()
        source_weights = self.critic.get_weights()
        new_weights = []
        for i in range(len(target_weights)):
            new_weights.append((1-self.tau)*target_weights[i] + self.tau*source_weights[i])
        self.target_critic.set_weights(new_weights)
    
    def save_agent(self, location):
        self.actor.save_model(location+"/actor_model.keras")
        self.target_actor.save_model(location+"/target_actor_model.keras")
        self.critic.save_model(location+"/critic_model.keras")
        self.target_critic.save_model(location+"/target_critic_model.keras")

    def load_agent(self, location):
        self.actor.load_model(location+"/actor_model.keras")
        self.target_actor.load_model(location+"/target_actor_model.keras")
        self.critic.load_model(location+"/critic_model.keras")
        self.target_critic.load_model(location+"/target_critic_model.keras")

    def print_position(self, episode):
         balance = self.current_state[-2]*self.environment.starting_balance
         shares = self.current_state[-1]*self.environment.shares_max
         portfolio = self.environment.portfolio_value
         timestep = self.environment.timestep
         print("action={0:<10.4f} balance={1:<10.4f} shares={2:<10.4f} portfolio={3:<10.4f} episode={4:<10d} timestep={5:<10d}".format(self.current_action[0], balance, shares, portfolio, episode, timestep))
