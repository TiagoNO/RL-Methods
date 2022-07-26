from platform import architecture
from cv2 import log
import gym
from RL_Methods.QLearning.QLearningAgent import QLearningAgent, DoubleQLearningAgent
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DDQN.DoubleDQNAgent import DoubleDQNAgent
from RL_Methods.PrioritizedDQN.PrioritizedDQN import PrioritizedDQN
from RL_Methods.DuelingDQN.DuelingDQNAgent import DuelingDQNAgent
from RL_Methods.Rainbow.RainbowAgent import RainbowAgent
from RL_Methods.MultiStepDQN.MultiStepDQNAgent import MultiStepDQNAgent
from RL_Methods.NoisyNetDQN.NoisyNetDQNAgent import NoisyNetDQNAgent
from RL_Methods.DistributionalDQN.DistributionalDQNAgent import DistributionalDQNAgent

import torch as th

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

    n_episodes = 20000
    plot_every = 100
    debug_every = 100
    env_name = "CartPole-v0"
    # env_name = "LunarLander-v2"

    # env = gym.make(env_name)
    # qAgent = QLearningAgent(env.action_space.n, lr=.5, discount=.99, initial_epsilon=1, final_epsilon=.1, epsilon_decay=1e-5)
    # qAgent.train(env, n_episodes)
    # generate_graph("tabular_qlearning", qAgent.scores, plot_every)

    # env = gym.make(env_name)
    # doubleQAgent = DoubleQLearningAgent(env.action_space.n, lr=.5, discount=.99, initial_epsilon=1, final_epsilon=.1 ,epsilon_decay=1e-5)
    # dq_scores = doubleQAgent.train(env, n_episodes)
    # generate_graph("tabular_double_qlearning", dq_scores, plot_every)

    env = gym.make(env_name)
    dqnAgent = DQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.1,
                        epsilon_decay=1e-5,
                        learning_rate=.001,
                        gamma=.9,
                        batch_size=64,
                        experience_buffer_size=1e5,
                        target_network_sync_freq=3000,
                        checkpoint_freq=50000,
                        savedir="experiments/dqn/",
                        log_freq=1,
                        architecture={'net_arch':[24, 24], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                            )
    dqnAgent.train(env, n_episodes)
    generate_graph(dqnAgent.savedir + "scores", dqnAgent.scores, plot_every)
    del env
    del dqnAgent

    env = gym.make(env_name)
    doubledqnAgent = DoubleDQNAgent(
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
                        checkpoint_freq=50000,
                        savedir="experiments/double_dqn/",
                        log_freq=1,
                        architecture={'net_arch':[24, 24], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                            )
    doubledqnAgent.train(env, n_episodes)
    generate_graph(doubledqnAgent.savedir + "scores", doubledqnAgent.scores, plot_every)
    del env
    del doubledqnAgent

    env = gym.make(env_name)
    prioritizedDQNAgent = PrioritizedDQN(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=.01,
                        gamma=.9,
                        batch_size=32,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        experience_prob_alpha=0.6,
                        experience_beta=0.4,
                        experience_beta_decay=1e-6,
                        checkpoint_freq=50000,
                        savedir="experiments/prioritized/",
                        log_freq=1,
                        architecture={'net_arch':[24, 24], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                        )
    prioritizedDQNAgent.train(env, n_episodes)
    generate_graph(prioritizedDQNAgent.savedir + "scores", prioritizedDQNAgent.scores, plot_every)
    del env
    del prioritizedDQNAgent
    
    env = gym.make(env_name)
    duelingDQNAgent = DuelingDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=.01,
                        gamma=.9,
                        batch_size=32,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        checkpoint_freq=50000,
                        savedir="",
                        log_freq=1,
                        architecture={'feature_arch':[24, 24], 'value_arch':[24, 12], 'advantage_arch':[24, 12], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                        )
    duelingDQNAgent.train(env, n_episodes)
    generate_graph(duelingDQNAgent.savedir + "scores", duelingDQNAgent.scores, plot_every)
    del env
    del duelingDQNAgent

    env = gym.make(env_name)
    multistepDQNAgent = MultiStepDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=.01,
                        gamma=.9,
                        batch_size=32,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        trajectory_steps=4,
                        checkpoint_freq=50000,
                        savedir="experiments/multistep/",
                        log_freq=1,
                        architecture={'net_arch':[24, 24], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                        )
    multistepDQNAgent.train(env, n_episodes)
    generate_graph(multistepDQNAgent.savedir + "scores", multistepDQNAgent.scores, plot_every)

    env = gym.make(env_name)
    noisyDQNAgent = NoisyNetDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=.01,
                        gamma=.9,
                        batch_size=32,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        sigma_init=.9,
                        checkpoint_freq=50000,
                        savedir="experiments/noisy/",
                        log_freq=1,
                        architecture={'net_arch':[24, 24], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                        )
    noisyDQNAgent.train(env, n_episodes)
    generate_graph(noisyDQNAgent.savedir + "scores", noisyDQNAgent.scores, plot_every)
    del env
    del noisyDQNAgent

    env = gym.make(env_name)
    distributionalDQNAgent = DistributionalDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=0.001,
                        gamma=.9,
                        batch_size=64,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        n_atoms=51,
                        min_value=1,
                        max_value=200,
                        checkpoint_freq=50000,
                        savedir="experiments/distributional/",
                        log_freq=1,
                        architecture={'net_arch':[24, 24], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                        )
    distributionalDQNAgent.train(env, n_episodes)
    generate_graph(distributionalDQNAgent.savedir + "scores", distributionalDQNAgent.scores, plot_every)
    del env
    del distributionalDQNAgent

    env = gym.make(env_name)
    rainbowAgent = RainbowAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        initial_epsilon=1.0,
                        final_epsilon=0.05,
                        epsilon_decay=1e-5,
                        learning_rate=.0001,
                        gamma=.9,
                        batch_size=32,
                        experience_buffer_size=1e6,
                        target_network_sync_freq=2000,
                        experience_prob_alpha=0.6,
                        experience_beta=0.4,
                        experience_beta_decay=1e-6,
                        trajectory_steps=4,
                        initial_sigma=.9,
                        n_atoms=2,
                        min_value=1,
                        max_value=200,
                        checkpoint_freq=10000,
                        savedir="experiments/rainbow/",
                        log_freq=1,
                        architecture={'feature_arch':[24, 24], 'value_arch':[24, 12], 'advantage_arch':[24, 12], 'activation_fn':th.nn.ReLU},
                        device='cpu'
                        )
    rainbowAgent.train(env, n_episodes)
    generate_graph(rainbowAgent.savedir + "scores", rainbowAgent.scores, plot_every)
    del env
    del rainbowAgent
    