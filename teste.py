import gym
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
    num_timesteps = 3000
    plot_every = 100
    debug_every = 100
    env_name = "CartPole-v0"
    
    # DQN models parameters
    epsilon = LinearSchedule(1.0, -1e-5, 0.05)
    learning_rate = LinearSchedule(0.001, -(0.001/num_timesteps), 0.0001)
    gamma = .9
    batch_size = 64
    experience_buffer_size = 1e6
    target_network_sync_freq = 2000

    #prioritized buffer parameters
    experience_prob_alpha = 0.6
    experience_beta = LinearSchedule(0.4, 1e-5, 1.0)

    # multi-step parameters
    trajectory_steps = 4

    # Noisy parameters
    initial_sigma = .9

    # Distributional parameters
    n_atoms = 2
    min_value = 1
    max_value = 200

    # Common parameters
    checkpoint_freq = 50000
    log_freq = 1
    arch = {'net_arch':[64, 64], 'activation_fn':th.nn.ReLU}
    dueling_arch = {'feature_arch':[64, 64], 'value_arch':[32], 'advantage_arch':[32], 'activation_fn':th.nn.ReLU}
    device = "cuda" if th.cuda.is_available() else "cpu"

    # env = gym.make(env_name)
    # dqnAgent = DQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=learning_rate,
    #                     epsilon=epsilon,
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     checkpoint_freq=checkpoint_freq,
    #                     savedir="experiments/dqn/",
    #                     log_freq=log_freq,
    #                     architecture=arch,
    #                     device=device,
    #                         )
    # dqnAgent.train(env, num_timesteps)
    # generate_graph(dqnAgent.savedir + "scores", dqnAgent.scores)
    # del env
    # del dqnAgent

    # learning_rate.reset()
    # epsilon.reset()
    # env = gym.make(env_name)
    # doubledqnAgent = DoubleDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=learning_rate,
    #                     epsilon=epsilon,
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     checkpoint_freq=checkpoint_freq,
    #                     savedir="experiments/double_dqn/",
    #                     log_freq=log_freq,
    #                     architecture=arch,
    #                     device=device,
    #                         )
    # doubledqnAgent.train(env, num_timesteps)
    # generate_graph(doubledqnAgent.savedir + "scores", doubledqnAgent.scores)
    # del env
    # del doubledqnAgent

    # learning_rate.reset()
    # epsilon.reset()
    # experience_beta.reset()    
    # env = gym.make(env_name)
    # prioritizedDQNAgent = PrioritizedDQN(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=learning_rate,
    #                     epsilon=epsilon,
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     experience_prob_alpha=experience_prob_alpha,
    #                     experience_beta=experience_beta,
    #                     checkpoint_freq=checkpoint_freq,
    #                     savedir="experiments/prioritized/",
    #                     log_freq=log_freq,
    #                     architecture=arch,
    #                     device=device,
    #                     )
    # prioritizedDQNAgent.train(env, num_timesteps)
    # generate_graph(prioritizedDQNAgent.savedir + "scores", prioritizedDQNAgent.scores)
    # del env
    # del prioritizedDQNAgent
    
    # learning_rate.reset()
    # epsilon.reset()    
    # env = gym.make(env_name)
    # duelingDQNAgent = DuelingDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=learning_rate,
    #                     epsilon=epsilon,
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     checkpoint_freq=checkpoint_freq,
    #                     savedir="experiments/dueling/",
    #                     log_freq=log_freq,
    #                     architecture=dueling_arch,
    #                     device=device,
    #                     )
    # duelingDQNAgent.train(env, num_timesteps)
    # generate_graph(duelingDQNAgent.savedir + "scores", duelingDQNAgent.scores)
    # del env
    # del duelingDQNAgent

    # learning_rate.reset()
    # epsilon.reset()
    # env = gym.make(env_name)
    # multistepDQNAgent = MultiStepDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=learning_rate,
    #                     epsilon=epsilon,
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     trajectory_steps=trajectory_steps,
    #                     checkpoint_freq=checkpoint_freq,
    #                     savedir="experiments/multistep/",
    #                     log_freq=log_freq,
    #                     architecture=arch,
    #                     device=device,
    #                     )
    # multistepDQNAgent.train(env, num_timesteps)
    # generate_graph(multistepDQNAgent.savedir + "scores", multistepDQNAgent.scores)

    # learning_rate.reset()
    # epsilon.reset()
    # env = gym.make(env_name)
    # noisyDQNAgent = NoisyNetDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=learning_rate,
    #                     epsilon=epsilon,
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     sigma_init=initial_sigma,
    #                     checkpoint_freq=checkpoint_freq,
    #                     savedir="experiments/noisy/",
    #                     log_freq=log_freq,
    #                     architecture=arch,
    #                     device=device,
    #                     )
    # noisyDQNAgent.train(env, num_timesteps)
    # generate_graph(noisyDQNAgent.savedir + "scores", noisyDQNAgent.scores)
    # del env
    # del noisyDQNAgent

    learning_rate.reset()
    epsilon.reset()
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
    generate_graph(distributionalDQNAgent.savedir + "scores", distributionalDQNAgent.scores)
    del env
    del distributionalDQNAgent

    learning_rate.reset()
    epsilon.reset()
    experience_beta.reset()
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
    generate_graph(rainbowAgent.savedir + "scores", rainbowAgent.scores)
    del env
    del rainbowAgent
    