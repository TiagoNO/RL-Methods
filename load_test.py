import gym
from RL_Methods.DQN import DQNAgent, Dueling, Noisy
from RL_Methods.DQN import DoubleDQNAgent
from RL_Methods.DQN import PrioritizedDQNAgent
from RL_Methods.DQN import DuelingDQNAgent
from RL_Methods.DQN import MultiStepDQNAgent
from RL_Methods.DQN import NoisyNetDQNAgent
from RL_Methods.DQN import DistributionalDQNAgent
from RL_Methods.DQN import RainbowAgent

from RL_Methods.utils.Callback import CheckpointCallback
from RL_Methods.utils.Logger import Logger

from RL_Methods.utils.Schedule import LinearSchedule

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
    th.set_num_threads(10)
    th.set_num_interop_threads(10)

    # num_timesteps = 60000
    # plot_every = 100
    # debug_every = 100
    env_name = "CartPole-v0"
    
    # # DQN models parameters
    # initial_epsilon = 1.0
    # final_epsilon = 0.05
    # epsilon_delta = -1e-5


    # initial_learning_rate = 0.0001
    # final_learning_rate = 0.00001
    # learning_rate_delta = -(initial_learning_rate/num_timesteps)

    # gamma = .9
    # batch_size = 128
    # experience_buffer_size = 1e6
    # target_network_sync_freq = 2000
    # grad_norm_clip=10

    # #prioritized buffer parameters
    # experience_prob_alpha = 0.6
    # initial_experience_beta = 0.4
    # final_experience_beta = 1.0
    # experience_beta_delta = (initial_experience_beta/num_timesteps)

    # # multi-step parameters
    # trajectory_steps = 4

    # # Noisy parameters
    # initial_sigma = 0.8

    # # Distributional parameters
    # n_atoms = 2
    # min_value = 1
    # max_value = 200

    # # Common parameters
    # checkpoint_freq = 50000
    # log_freq = 1
    # arch = {'net_arch':[64, 64], 'activation_fn':th.nn.ReLU}
    # dueling_arch = {'feature_arch':[64, 64], 'value_arch':[32], 'advantage_arch':[32], 'activation_fn':th.nn.ReLU}
    # device = "cuda" if th.cuda.is_available() else "cpu"

    # env = gym.make(env_name)
    # dqnAgent = DQNAgent.load("experiments/dqn/dqn_last.zip")
    # dqnAgent.test(env, 10)
    # del env
    # del dqnAgent

    # env = gym.make(env_name)
    # distributionalAgent = DistributionalDQNAgent.load("experiments/distributional/dqn_last.zip")
    # distributionalAgent.test(env, 10)
    # del env
    # del distributionalAgent

    # env = gym.make(env_name)
    # doubleAgent = DoubleDQNAgent.load("experiments/double/dqn_last.zip")
    # doubleAgent.test(env, 10)
    # del env
    # del doubleAgent

    # env = gym.make(env_name)
    # duelingAgent = DuelingDQNAgent.load("experiments/dueling/dqn_last.zip")
    # duelingAgent.test(env, 10)
    # del env
    # del duelingAgent

    # env = gym.make(env_name)
    # multistepAgent = MultiStepDQNAgent.load("experiments/multistep/dqn_last.zip")
    # multistepAgent.test(env, 10)
    # del env
    # del multistepAgent

    # env = gym.make(env_name)
    # noisyAgent = NoisyNetDQNAgent.load("experiments/noisy/dqn_last.zip")
    # noisyAgent.test(env, 10)
    # del env
    # del noisyAgent

    # env = gym.make(env_name)
    # prioritizedAgent = PrioritizedDQNAgent.load("experiments/prioritized/dqn_last.zip")
    # prioritizedAgent.test(env, 10)
    # del env
    # del prioritizedAgent

    env = gym.make(env_name)
    rainbowAgent = RainbowAgent.load("experiments/rainbow/rainbow_100000_steps.zip")
    rainbowAgent.test(env, 10)
    del env
    del rainbowAgent    
