import numpy as np
import pickle 
import gym
resume = True
render = True

# note that a single episode in pong refers to a single game (21 points to win)
# additionally, a batch refers to a set of (batch_size) episodes after which weights get updated

''' note the model structure with the following layers:
    Layer 0 (a0): 80*80=6400 input neurons
    Layer 1 (a1): 200 hidden neurons; activation=relu
    Layer 2 (a2): 1 output neuron; activation=sigmoid

    a0 -----> a1 -----> a2
'''

# define model hyperparameters
n_input = 80*80
n_hidden = 200 # number of hidden neurons
discount_rate = 0.99 # discount rate for computing discounted reward prior to the episode terminal reward in pong
decay_rate = 0.99 # decay rate for rmsprop
learning_rate = 10**-4 # learning rate for rmsprop
batch_size = 10 # batch size for applying rmsprop

# initialize model parameters, caches
if resume:
    parameters = pickle.load(open('../trained_parameters/trained_parameters.p', 'rb'))
else:
    parameters = {"W1": np.random.random((n_hidden, n_input))/np.sqrt(n_input), # Xavier initialization
              "W2": np.random.random((1, n_hidden))/np.sqrt(n_hidden) # Xavier initialization
              }
activations_cache = {"a1":[],
                     "a2":[]} # stores the activations for layers 1 and 2 for an entire episode in two lists of activations for each frame in the episode

gradients_cache = {"dW1": 0,
                   "dW2": 0} # stores the sum of gradients over all episodes in a batch as a dictionary

rmsprop_cache = {"dW1": 0,
                 "dW2": 0} # stores the moving average squared gradients over multiple batches in a dictionary

def preprocess(observation):
    # initial images are (210 x 160 x 3)
    processed_observation = observation[35:195] # crop the rows to exclude the scoreboard and redundant bottom area
    processed_observation = processed_observation[::2, ::2, 0] # downsampling by a factor 2 and retaining only the red channel
    processed_observation[processed_observation==144] = 0 # converting to grayscale by converting R=144 (part of bg) to black
    processed_observation[processed_observation==109] = 0 # converting to grayscale by converting R=109 (part of bg) to black
    processed_observation[processed_observation!=0] = 1 # remaining pixels become white
    return processed_observation.astype(np.float64).reshape(-1, 1) # returns a 1600 x 1 array

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    x[x<0]=0
    return x

def policy_propagate(x):
    a0 = x
    z1 = np.dot(parameters["W1"], a0)
    a1 = relu(z1)
    z2 = np.dot(parameters["W2"], a1)
    a2 = sigmoid(z2)
    return (a1, a2)

def discount_rewards(reward_sequence):
    discounted_reward_sequence = np.zeros_like(reward_sequence)
    reward_momentum=0
    for i in reversed(range(len(reward_sequence))):
        if reward_sequence[i] != 0:
            reward_momentum=0
        reward_momentum = discount_rate*reward_momentum + reward_sequence[i]
        discounted_reward_sequence[i] = reward_momentum
    discounted_reward_sequence -= np.mean(discounted_reward_sequence)
    discounted_reward_sequence /= np.std(discounted_reward_sequence)

    return discounted_reward_sequence

def compute_gradients(x_list, y, discounted_reward_sequence, parameters):
    x_list = np.array(x_list)
    y = np.array(y).reshape(-1, 1)
    a1 = np.array(activations_cache['a1'])
    a2 = np.array(activations_cache['a2']).reshape(-1, 1)
    dz2 = np.multiply(a2-y, discounted_reward_sequence.reshape(-1, 1))
    dW2 = np.tensordot(dz2.T, a1, (1, 0)).reshape(1, n_hidden)
    da1 = np.tensordot(parameters['W2'].T, dz2.T, (-1, 0))
    dz1 = da1*(a1.T>0)
    dW1 = np.matmul(dz1, x_list).reshape(n_hidden, n_input)
    return (dW1, dW2)

def update_parameters(parameters, gradients_cache):
    epsilon = 10**-5
    rmsprop_cache['dW1'] = rmsprop_cache['dW1']*decay_rate + (gradients_cache['dW1']**2)*(1-decay_rate)
    rmsprop_cache['dW2'] = rmsprop_cache['dW2']*decay_rate + (gradients_cache['dW2']**2)*(1-decay_rate)
    parameters['W1'] -= learning_rate*gradients_cache['dW1']/(np.sqrt(rmsprop_cache['dW1'])+epsilon)
    parameters['W2'] -= learning_rate*gradients_cache['dW2']/(np.sqrt(rmsprop_cache['dW2'])+epsilon)
    gradients_cache['dW1', 'dW2'] = 0 # reset the gradients_cache since a batch is over 

def play():
    env = gym.make("Pong-v0", render_mode="human" if render else None)
    observation = env.reset()
    observation = observation[0]
    prev_processed_observation = None
    won = 0
    games = 0
    positive_reward_sum = 0
    while True:
        processed_observation = preprocess(observation)
        if prev_processed_observation is None:
            x = processed_observation
        else:
            x = processed_observation-prev_processed_observation
        prev_processed_observation = processed_observation
        a1, action_prob = policy_propagate(x) # compute the action probability using the current policy network
        action = 2 if np.random.random()<action_prob else 3 # sampling from the action distribution; 2 and 3 correspond to UP and DOWN movement respectively in the OpenAI Gym 
        observation, reward, terminated, truncated, _ = env.step(action) # use the OpenAI Gym to play one frame of Pong
        positive_reward_sum = positive_reward_sum + reward if (reward > 0) else positive_reward_sum
        if (terminated or truncated):
            if positive_reward_sum==21:
                won+=1
            games += 1
            print(f"Won {won} games out of {games}")
            positive_reward_sum=0
            env.reset()

def train():
    env = gym.make("Pong-v0", render_mode="human" if render else None)
    observation = env.reset()
    observation = observation[0] # The 0th component of the observation tuple is the real data
    prev_processed_observation = None
    x_list = []
    y_labels = []
    reward_sum = 0
    reward_sequence = []
    reward_momentum = None
    episodes = 0

    while True:
        if render: env.render()
        processed_observation = preprocess(observation) 
        if prev_processed_observation is None:
            x = processed_observation
        else:
            x = processed_observation - prev_processed_observation
        prev_processed_observation = processed_observation
        a1, action_prob = policy_propagate(x) # compute the action probability using the current policy network
        action = 2 if np.random.random()<action_prob else 3 # sampling from the action distribution; 2 and 3 correspond to UP and DOWN movement respectively in the OpenAI Gym 
        
        # now we need to generate some labels using the sampled actions because we don't have training data in RL
        y = 1 if action==2 else 0
        # cross_entropy_loss (negative log-prob loss) will be minimized (corresponds to maximizing log-likelihood)

        x_list.append(x.ravel())
        activations_cache['a1'].append(a1)
        activations_cache['a2'].append(action_prob)
        y_labels.append(y)


        observation, reward, terminated, truncated, info = env.step(action) # use the OpenAI Gym to play one frame of Pong
        reward_sum += reward
        reward_sequence.append(reward)
        done = terminated or truncated # indicates the end of an episode (21 points)

        if done:
            episodes += 1
            discounted_reward_sequence = discount_rewards(reward_sequence)
            gradients = compute_gradients(x_list, y_labels, discounted_reward_sequence, parameters)
            gradients_cache['dW1']+=gradients[0]
            gradients_cache['dW2']+=gradients[1]

            if episodes%batch_size == 0:
                update_parameters(parameters, gradients_cache)
                print("Finished batch {0}".format(episodes//batch_size))
                print(parameters['W1'][0, 0])

            reward_momentum = reward_sum if reward_momentum == None else reward_momentum*0.99 + reward_sum*0.01
            print("Finished episode {0} with reward_sum={1} and reward_momentum={2}".format(episodes, reward_sum, reward_momentum))
            if episodes%10 == 0: pickle.dump(parameters, open('../trained_parameters/trained_parameters.p', 'wb'))
            x_list = []
            y_labels = []
            reward_sum = 0
            reward_sequence = []
            activations_cache = {"a1":[],
                                "a2":[]}
            observation = env.reset() 
            observation = observation[0] # The 0th component of the observation tuple is the real data

play()