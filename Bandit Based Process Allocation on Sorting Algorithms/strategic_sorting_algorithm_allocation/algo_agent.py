import numpy as np
import time 

class Algo_Agent():
    def __init__(self):
        self.n = 0

    def epsilon_greedy(self, action_values, epsilon):
        if np.random.rand() > epsilon:
            # select greedy action (exploit)
            return np.argmin(action_values)
        else:
            # explore
            num_actions = action_values.shape[0]
            return np.random.choice(list(range(num_actions)))
        
    def run(self, num_actions, initial_action_values, num_tasks, arr_size, algorithms, epsilon):
        """ step size only to be implemented with moving average cost formulation """
        action_values = initial_action_values.copy()
        self.average_cost_series = np.zeros(num_tasks)
        self.visitations = np.zeros(num_actions)

        self.n = 0
        for i in range(num_tasks):
            self.n += 1

            unsorted = np.random.rand(arr_size)
            action = self.epsilon_greedy(action_values, epsilon)
            self.visitations[action] += 1
            start = time.time_ns()
            sorted = algorithms[action](unsorted, 0, arr_size-1)
            end = time.time_ns()
            cost = (end-start)/(1e6)
            action_values[action] = action_values[action] + (cost-action_values[action])/self.visitations[action]
            self.average_cost_series[i] = self.average_cost_series[i-1] + (cost-self.average_cost_series[i-1])/self.n

        return self.average_cost_series, self.visitations
    
    def test_allocation(self, num_actions, initial_action_values, num_tasks, num_runs, arr_size, algorithms, epsilon):
        
        costs_matrix = np.zeros(shape=(num_runs, num_tasks)) # stores the running simple mean cost over the past steps for each time step for each run
        visitation_distribution = np.zeros(shape=num_actions)

        for i in range(num_runs):
            average_cost_series, visitations = self.run(num_actions, initial_action_values, num_tasks, arr_size, algorithms, epsilon)
            costs_matrix[i] = average_cost_series
            for k in range(num_actions):
                if visitations[k] > 0:
                    visitation_distribution[k] = visitation_distribution[k] + (visitations[k]-visitation_distribution[k])/(i+1)

        cost_mean_history = costs_matrix.mean(axis=0) # stores the mean cost across runs for every timestep
        cost_std_history = costs_matrix.std(axis=0) # stores the cost standard deviation across runs for every timestep

        return cost_mean_history, cost_std_history, visitation_distribution