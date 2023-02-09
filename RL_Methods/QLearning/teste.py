import gymnasium as gym
from RL_Methods.QLearning import QLearningAgent, DoubleQLearningAgent

from RL_Methods.utils.Callback import CheckpointCallback
from RL_Methods.utils.Logger import Logger

from RL_Methods.utils.Schedule import LinearSchedule, ConstantSchedule


import torch as th

import matplotlib.pyplot as plt
import numpy as np

def generate_graph(filename, scores):
    avg = []
    for i in range(len(scores)):
        avg.append(np.mean(scores[max(0, i-100):i]))

    plt.title("Learning curve in trainning")
    plt.xlabel("Episodes")
    plt.title("Average reward last 100 episodes")
    plt.plot(avg)
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':

    num_timesteps = 1000000
    plot_every = 100
    debug_every = 100
    env_name = "CartPole-v1"
    
    # DQN models parameters
    initial_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_delta = -1e-5

    initial_learning_rate = 0.9
    final_learning_rate = 0.01
    learning_rate_delta = -(initial_learning_rate/num_timesteps)

    gamma = .9

    # Common parameters
    checkpoint_freq = 5000
    log_freq = 1

    env = gym.make(env_name)
    qAgent = QLearningAgent(
                        env.action_space.n,
                        learning_rate=ConstantSchedule(initial_learning_rate),
                        discount=gamma,
                        epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
                        logger=Logger("experiments/dqn/log_file"),                        
                        save_log_every=100,
                        verbose=3
                        )
    qAgent.train(env, num_timesteps)
    #generate_graph(dqnAgent.logger.directory + "scores", dqnAgent.data['scores'])
    del env
    del qAgent

    env = gym.make(env_name)
    doubleqAgent = DoubleQLearningAgent(
                        env.action_space.n,
                        learning_rate=ConstantSchedule(initial_learning_rate),
                        discount=gamma,
                        epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
                        logger=Logger("experiments/double/log_file"),
                        save_log_every=100,
                        verbose=3
                        )
    doubleqAgent.train(env, num_timesteps)
    #generate_graph(doubledqnAgent.logger.directory + "scores", doubledqnAgent.data['scores'])
    del env
    del doubleqAgent



    