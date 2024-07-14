import numpy as np

class Bandit():
    def __init__(self):
        self.n = 0 
        self.visitations = None
        self.executed_optimal = None

    def greedy(self, action_values):
        return np.argmax(action_values)

    def epsilon_greedy(self, action_values, epsilon):
        if np.random.rand() > epsilon:
            # select greedy action (exploit)
            return np.argmax(action_values)
        else:
            # explore
            num_actions = action_values.shape[0]
            return np.random.choice(list(range(num_actions)))
    
    def upper_confidence_bound(self, action_values, exploration):
        return np.argmax(action_values+exploration*np.sqrt(np.log(self.n)/(self.visitations+1e-7)))

    def choose_policy(self, policy):
        if policy == 'greedy':
            return  self.greedy
        elif policy =='epsilon_greedy':
            return self.epsilon_greedy
        elif policy == 'upper_confidence_bound':
            return self.upper_confidence_bound
        
    def run(self, num_actions, arms_mean, arms_spread, standard_deviations, initial_action_values, num_timesteps, policy, policy_args, reward_formulation='simple_average', step_size=None):
        """ step size only to be implemented with moving average reward formulation """
        action_values = initial_action_values.copy()
        self.average_reward_series = np.zeros(num_timesteps)
        self.visitations = np.zeros(num_actions)
        self.executed_optimal = np.zeros(num_timesteps)
        optimal_action_values = np.random.normal(loc=arms_mean, scale=arms_spread, size=num_actions)

        self.n = 0
        for i in range(num_timesteps):
                self.n += 1
                action = policy(action_values, *policy_args)
                if action==np.argmax(optimal_action_values):
                    self.executed_optimal[i] = 1
                self.visitations[action] += 1
                reward = np.random.normal(loc=optimal_action_values[action], scale=standard_deviations[action])
                if reward_formulation == 'simple_average':
                    action_values[action] = action_values[action] + (reward-action_values[action])/self.visitations[action]
                    self.average_reward_series[i] = self.average_reward_series[i-1] + (reward-self.average_reward_series[i-1])/self.n
                elif reward_formulation == 'moving_average':
                    action_values[action] = action_values[action] + step_size*(reward-action_values[action])
                    self.average_reward_series[i] = self.average_reward_series[i-1] + step_size*(reward-self.average_reward_series[i-1])

        return self.average_reward_series, self.visitations, self.executed_optimal
        
    def test_stationary(self, num_actions, arms_mean, arms_spread, standard_deviations, initial_action_values, num_timesteps, num_runs, policy, policy_args, reward_formulation='simple_average', step_size=None):
        
        policy = self.choose_policy(policy)
        rewards_matrix = np.zeros(shape=(num_runs, num_timesteps)) # stores the running simple mean reward over the past steps for each time step for each run
        visitation_distribution = np.zeros(shape=num_actions)
        optimal_executions_distribution = np.zeros(shape=num_timesteps)

        for i in range(num_runs):
            average_reward_series, visitations, executed_optimal = self.run(num_actions, arms_mean, arms_spread, standard_deviations, initial_action_values, num_timesteps, policy, policy_args, reward_formulation=reward_formulation, step_size=step_size)
            rewards_matrix[i] = average_reward_series
            for k in range(num_actions):
                if visitations[k] > 0:
                    visitation_distribution[k] = visitation_distribution[k] + (visitations[k]-visitation_distribution[k])/(i+1)
            optimal_executions_distribution = optimal_executions_distribution + executed_optimal

        reward_mean_history = rewards_matrix.mean(axis=0) # stores the mean reward across runs for every timestep
        reward_std_history = rewards_matrix.std(axis=0) # stores the reward standard deviation across runs for every timestep

        return reward_mean_history, reward_std_history, visitation_distribution, optimal_executions_distribution
    
    def test_nonstationary(self, num_actions, arms_mean, arms_spread, standard_deviations, drift, initial_action_values, num_timesteps, num_runs, policy, policy_args, reward_formulation='simple_average', step_size=None):
        
        policy = self.choose_policy(policy)
        rewards_matrix = np.zeros(shape=(num_runs, num_timesteps)) # stores the running simple mean reward over the past steps for each time step for each run
        visitation_distribution = np.zeros(shape=num_actions)
        optimal_executions_distribution = np.zeros(shape=num_timesteps)

        for i in range(num_runs):
            average_reward_series, visitations, executed_optimal = self.run(num_actions, arms_mean, arms_spread, standard_deviations, initial_action_values, num_timesteps, policy, policy_args, reward_formulation=reward_formulation, step_size=step_size)
            rewards_matrix[i] = average_reward_series
            for k in range(num_actions):
                if visitations[k] > 0:
                    visitation_distribution[k] = visitation_distribution[k] + (visitations[k]-visitation_distribution[k])/(i+1)
            optimal_executions_distribution = optimal_executions_distribution + executed_optimal
            arms_mean = arms_mean + drift*np.random.normal(size=num_actions)
            
        reward_mean_history = rewards_matrix.mean(axis=0) # stores the mean reward across runs for every timestep
        reward_std_history = rewards_matrix.std(axis=0) # stores the reward standard deviation across runs for every timestep

        return reward_mean_history, reward_std_history, visitation_distribution, optimal_executions_distribution

class Gradient_Bandit():
    def __init__(self):
        self.n = 0

    def run(self, num_actions, arms_mean, arms_spread, standard_deviations, num_timesteps, step_size, use_baseline=True):
        """ step size only to be implemented with moving average reward formulation """
        action_preferences = np.zeros(num_actions)
        self.average_reward_series = np.zeros(num_timesteps)
        self.visitations = np.zeros(num_actions)
        optimal_action_values = np.random.normal(loc=arms_mean, scale=arms_spread, size=num_actions)
        self.executed_optimal = np.zeros(num_timesteps)

        self.n = 0
        for i in range(num_timesteps):
            self.n += 1
            probabilities = np.exp(action_preferences)/np.sum(np.exp(action_preferences))
            cumulative_probabilities = np.cumsum(probabilities)
            temp = np.random.rand()
            j = 0
            while temp > cumulative_probabilities[j]:
                j += 1
            action = j
            if action==np.argmax(optimal_action_values):
                self.executed_optimal[i] = 1
            self.visitations[action] += 1
            reward = np.random.normal(loc=optimal_action_values[action], scale=standard_deviations[action])
            if use_baseline:
                action_preferences = action_preferences + step_size*(reward-self.average_reward_series[self.n-1])*((np.arange(num_actions)==action).astype("uint8")-probabilities)
            else:
                action_preferences = action_preferences + step_size*(reward)*((np.arange(num_actions)==action).astype("uint8")-probabilities)
            self.average_reward_series[i] = self.average_reward_series[i-1] + (reward-self.average_reward_series[i-1])/(self.n)

        return self.average_reward_series, self.visitations, self.executed_optimal

    def test_stationary(self, num_actions, arms_mean, arms_spread, standard_deviations, num_timesteps, num_runs, step_size, use_baseline=True):
        
        rewards_matrix = np.zeros(shape=(num_runs, num_timesteps)) # stores the running simple mean reward over the past steps for each time step for each run
        visitation_distribution = np.zeros(shape=num_actions)
        optimal_executions_distribution = np.zeros(shape=num_timesteps)

        for i in range(num_runs):
            average_reward_series, visitations, executed_optimal = self.run(num_actions, arms_mean, arms_spread, standard_deviations, num_timesteps, step_size, use_baseline=use_baseline)
            rewards_matrix[i] = average_reward_series
            for k in range(num_actions):
                if visitations[k] > 0:
                    visitation_distribution[k] = visitation_distribution[k] + (visitations[k]-visitation_distribution[k])/(i+1)
            optimal_executions_distribution = optimal_executions_distribution + executed_optimal

        reward_mean_history = rewards_matrix.mean(axis=0) # stores the mean reward across runs for every timestep
        reward_std_history = rewards_matrix.std(axis=0) # stores the reward standard deviation across runs for every timestep

        return reward_mean_history, reward_std_history, visitation_distribution, optimal_executions_distribution