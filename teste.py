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

from RL_Methods.utils.Schedule import LinearSchedule

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

    num_timesteps = 300000
    plot_every = 100
    debug_every = 100
    env_name = "CartPole-v0"
    
    # DQN models parameters
    epsilon = LinearSchedule(1.0, -1e-5, 0.05)
    learning_rate = LinearSchedule(0.01, -(0.01/num_timesteps), 0.0001)
    gamma=.9
    batch_size=64
    experience_buffer_size=1e6
    target_network_sync_freq=2000

    #prioritized buffer parameters
    experience_prob_alpha=0.6
    experience_beta = LinearSchedule(0.4, 1e-5, 1.0)

    # multi-step parameters
    trajectory_steps=4

    # Noisy parameters
    initial_sigma=.9

    # Distributional parameters
    n_atoms=2
    min_value=1
    max_value=200

    # Common parameters
    checkpoint_freq=50000
    savedir="experiments/rainbow/"
    log_freq=1
    arch={'net_arch':[24, 24, 24], 'activation_fn':th.nn.ReLU}
    dueling_arch={'feature_arch':[24, 24, 24], 'value_arch':[24, 12], 'advantage_arch':[24, 12], 'activation_fn':th.nn.ReLU}
    device="cpu"

    env = gym.make(env_name)
    dqnAgent = DQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/dqn/",
                        log_freq=log_freq,
                        architecture=arch,
                        device=device,
                            )
    dqnAgent.train(env, num_timesteps)
    generate_graph(dqnAgent.savedir + "scores", dqnAgent.scores, plot_every)
    del env
    del dqnAgent

    env = gym.make(env_name)
    doubledqnAgent = DoubleDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/double_dqn/",
                        log_freq=log_freq,
                        architecture=arch,
                        device=device,
                            )
    doubledqnAgent.train(env, num_timesteps)
    generate_graph(doubledqnAgent.savedir + "scores", doubledqnAgent.scores, plot_every)
    del env
    del doubledqnAgent

    env = gym.make(env_name)
    prioritizedDQNAgent = PrioritizedDQN(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        experience_prob_alpha=experience_prob_alpha,
                        experience_beta=experience_beta,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/prioritized/",
                        log_freq=log_freq,
                        architecture=arch,
                        device=device,
                        )
    prioritizedDQNAgent.train(env, num_timesteps)
    generate_graph(prioritizedDQNAgent.savedir + "scores", prioritizedDQNAgent.scores, plot_every)
    del env
    del prioritizedDQNAgent
    
    env = gym.make(env_name)
    duelingDQNAgent = DuelingDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/dueling/",
                        log_freq=log_freq,
                        architecture=dueling_arch,
                        device=device,
                        )
    duelingDQNAgent.train(env, num_timesteps)
    generate_graph(duelingDQNAgent.savedir + "scores", duelingDQNAgent.scores, plot_every)
    del env
    del duelingDQNAgent

    env = gym.make(env_name)
    multistepDQNAgent = MultiStepDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        trajectory_steps=trajectory_steps,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/multistep/",
                        log_freq=log_freq,
                        architecture=arch,
                        device=device,
                        )
    multistepDQNAgent.train(env, num_timesteps)
    generate_graph(multistepDQNAgent.savedir + "scores", multistepDQNAgent.scores, plot_every)

    env = gym.make(env_name)
    noisyDQNAgent = NoisyNetDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        sigma_init=initial_sigma,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/noisy/",
                        log_freq=log_freq,
                        architecture=arch,
                        device=device,
                        )
    noisyDQNAgent.train(env, num_timesteps)
    generate_graph(noisyDQNAgent.savedir + "scores", noisyDQNAgent.scores, plot_every)
    del env
    del noisyDQNAgent

    env = gym.make(env_name)
    distributionalDQNAgent = DistributionalDQNAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        n_atoms=n_atoms,
                        min_value=min_value,
                        max_value=max_value,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/distributional/",
                        log_freq=log_freq,
                        architecture=arch,
                        device=device,
                        )
    distributionalDQNAgent.train(env, num_timesteps)
    generate_graph(distributionalDQNAgent.savedir + "scores", distributionalDQNAgent.scores, plot_every)
    del env
    del distributionalDQNAgent

    env = gym.make(env_name)
    rainbowAgent = RainbowAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        experience_prob_alpha=experience_prob_alpha,
                        experience_beta=experience_beta,
                        trajectory_steps=trajectory_steps,
                        initial_sigma=initial_sigma,
                        n_atoms=n_atoms,
                        min_value=min_value,
                        max_value=max_value,
                        checkpoint_freq=checkpoint_freq,
                        savedir="experiments/rainbow/",
                        log_freq=log_freq,
                        architecture=dueling_arch,
                        device=device,
                        )
    rainbowAgent.train(env, num_timesteps)
    generate_graph(rainbowAgent.savedir + "scores", rainbowAgent.scores, plot_every)
    del env
    del rainbowAgent
    