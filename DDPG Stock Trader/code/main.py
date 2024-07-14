from agent import DDPG_Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

training = False
load_weights = True
save_weights = False

current_working_directory = os.getcwd()
parent_path = str(Path(current_working_directory).parent.absolute())
location = parent_path+"/model_saves"

agent = DDPG_Agent(3+3+2, 1)

if load_weights and os.path.exists(location) and len(os.listdir(location))>0:
    agent.load_agent(location)
    print("Loaded agent from " + location)

if (load_weights or save_weights) and not os.path.exists(location):
    os.mkdir(location)

episodes = 1
discounted_reward = 0

for episode in range(episodes):
    end_of_episode = False
    while not end_of_episode:
        end_of_episode = agent.train_step(training, episode)
    if training and save_weights:
        agent.save_agent(location)

portfolio_values = agent.environment.portfolio_values
plt.plot(portfolio_values, label='portfolio')
market_value = agent.environment.indices['Close']*(agent.environment.indices_max['Close']-agent.environment.indices_min['Close']) + agent.environment.indices_min['Close']
market_value = market_value.to_numpy()*agent.environment.starting_balance/(market_value.iloc[0])

plt.plot((np.repeat(np.expand_dims(market_value, axis=0), repeats=episodes, axis=0)).flatten(), label='market')
plt.legend()
plt.show()