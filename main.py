import gym
from numpy.lib.function_base import sort_complex
from traitlets import observe
from QLearning.QLearningAgent import QLearningAgent, DoubleQLearningAgent
from NNMethods.DQN.DQNAgent import DQNAgent
from NNMethods.DDQN.DoubleDQNAgent import DoubleDQNAgent
import matplotlib.pyplot as plt
import numpy as np

def generate_graph(filename, scores, plot_every=10):
    avg = []
    x_sz = len(scores)
    for i in range(0, x_sz, plot_every):
        j = min(i+plot_every, x_sz)
        mean = np.mean(scores[i:j])
        avg.append(mean)

    plt.plot(range(len(avg)), avg)
    plt.savefig(filename)

if __name__ == '__main__':
    file_name = "montecarloagent.pk"

    n_episodes = 100000
    plot_every = 100
    debug_every = 100
    env_name = "CartPole-v0"


    env = gym.make(env_name)

    # qAgent = QLearningAgent(env.action_space.n, lr=.5, discount=.99, initial_epsilon=1, final_epsilon=.1, epsilon_decay=1e-5)
    # q_scores = qAgent.train(env, n_episodes)
    
    # env = gym.make(env_name)

    # doubleQAgent = DoubleQLearningAgent(env.action_space.n, lr=.5, discount=.99, initial_epsilon=1, final_epsilon=.1 ,epsilon_decay=1e-5)
    # dq_scores = doubleQAgent.train(env, n_episodes)

    # env = gym.make(env_name)
    # deepqAgent = DQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     initial_epsilon=1.0,
    #                     final_epsilon=0.05,
    #                     epsilon_decay=1e-5,
    #                     learning_rate=.001,
    #                     gamma=.9,
    #                     batch_size=32,
    #                     experience_buffer_size=1e6,
    #                     target_network_sync_freq=2000,
    #                     device='cpu'
    #                         )
    # deepq_scores = deepqAgent.train(env, n_episodes)

    env = gym.make(env_name)
    deepqAgent = DoubleDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=.001,
                        gamma=.9,
                        batch_size=32,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        device='cpu'
                            )
    doubledeepq_scores = deepqAgent.train(env, n_episodes)


    # generate_graph("figures1", q_scores, plot_every)
    # generate_graph("figures2", dq_scores, plot_every)
    generate_graph("figures3", doubledeepq_scores, plot_every)
