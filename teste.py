from tabnanny import check
import gym
from RL_Methods.DQN import DQNAgent
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
    th.set_num_threads(5)
    th.set_num_interop_threads(5)

    num_timesteps = 100000
    plot_every = 100
    debug_every = 100
    env_name = "CartPole-v0"
    
    # DQN models parameters
    initial_epsilon = 1.0
    final_epsilon = 0.05
    epsilon_delta = -1e-5


    initial_learning_rate = 0.0001
    final_learning_rate = 0.00001
    learning_rate_delta = -(initial_learning_rate/num_timesteps)

    gamma = .9
    batch_size = 128
    experience_buffer_size = 1e6
    target_network_sync_freq = 2000
    grad_norm_clip=10

    #prioritized buffer parameters
    experience_prob_alpha = 0.6
    initial_experience_beta = 0.4
    final_experience_beta = 1.0
    experience_beta_delta = (initial_experience_beta/num_timesteps)

    # multi-step parameters
    trajectory_steps = 4

    # Noisy parameters
    initial_sigma = 0.5

    # Distributional parameters
    n_atoms = 2
    min_value = 1
    max_value = 200

    # Common parameters
    checkpoint_freq = 5000
    log_freq = 1
    arch = {'net_arch':[64, 64], 'activation_fn':th.nn.ReLU}
    dueling_arch = {'feature_arch':[64, 64], 'value_arch':[32], 'advantage_arch':[32], 'activation_fn':th.nn.ReLU}
    device = "cuda" if th.cuda.is_available() else "cpu"

    # env = gym.make(env_name)
    # dqnAgent = DQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=arch,
    #                     callbacks=CheckpointCallback("experiments/dqn/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/dqn/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # dqnAgent.train(env, num_timesteps)
    # generate_graph(dqnAgent.logger.directory + "scores", dqnAgent.data['scores'])
    # del env
    # del dqnAgent

    # env = gym.make(env_name)
    # doubledqnAgent = DoubleDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=arch,
    #                     callbacks=CheckpointCallback("experiments/double/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/double/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # doubledqnAgent.train(env, num_timesteps)
    # generate_graph(doubledqnAgent.logger.directory + "scores", doubledqnAgent.data['scores'])
    # del env
    # del doubledqnAgent


    # env = gym.make(env_name)
    # prioritizedDQNAgent = PrioritizedDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     experience_prob_alpha=experience_prob_alpha,
    #                     experience_beta=LinearSchedule(initial_experience_beta, experience_beta_delta, final_experience_beta),
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=arch,
    #                     callbacks=CheckpointCallback("experiments/prioritized/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/prioritized/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # prioritizedDQNAgent.train(env, num_timesteps)
    # generate_graph(prioritizedDQNAgent.logger.directory + "scores", prioritizedDQNAgent.data['scores'])
    # del env
    # del prioritizedDQNAgent
    
    # env = gym.make(env_name)
    # duelingDQNAgent = DuelingDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=dueling_arch,
    #                     callbacks=CheckpointCallback("experiments/dueling/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/dueling/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # duelingDQNAgent.train(env, num_timesteps)
    # generate_graph(duelingDQNAgent.logger.directory + "scores", duelingDQNAgent.data['scores'])
    # del env
    # del duelingDQNAgent

    # env = gym.make(env_name)
    # multistepDQNAgent = MultiStepDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     trajectory_steps=trajectory_steps,
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=arch,
    #                     callbacks=CheckpointCallback("experiments/multistep/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/multistep/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # multistepDQNAgent.train(env, num_timesteps)
    # generate_graph(multistepDQNAgent.logger.directory + "scores", multistepDQNAgent.data['scores'])
    # del env
    # del multistepDQNAgent

    # env = gym.make(env_name)
    # noisyDQNAgent = NoisyNetDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     sigma_init=initial_sigma,
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=arch,
    #                     callbacks=CheckpointCallback("experiments/noisy/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/noisy/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # noisyDQNAgent.train(env, num_timesteps)
    # generate_graph(noisyDQNAgent.logger.directory + "scores", noisyDQNAgent.data['scores'])
    # del env
    # del noisyDQNAgent

    # env = gym.make(env_name)
    # distributionalDQNAgent = DistributionalDQNAgent(
    #                     env.observation_space.shape,
    #                     env.action_space.n,
    #                     learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
    #                     epsilon=LinearSchedule(initial_epsilon, epsilon_delta, final_epsilon),
    #                     gamma=gamma,
    #                     batch_size=batch_size,
    #                     experience_buffer_size=experience_buffer_size,
    #                     target_network_sync_freq=target_network_sync_freq,
    #                     n_atoms=n_atoms,
    #                     min_value=min_value,
    #                     max_value=max_value,
    #                     grad_norm_clip=grad_norm_clip,
    #                     architecture=arch,
    #                     callbacks=CheckpointCallback("experiments/distributional/", "dqn", checkpoint_freq),
    #                     logger=Logger("experiments/distributional/log_file"),
    #                     log_freq=10,
    #                     device=device
    #                     )
    # distributionalDQNAgent.train(env, num_timesteps)
    # generate_graph(distributionalDQNAgent.logger.directory + "scores", distributionalDQNAgent.data['scores'])
    # del env
    # del distributionalDQNAgent

    env = gym.make(env_name)
    rainbowAgent = RainbowAgent(
                        env.observation_space.shape,
                        env.action_space.n,
                        learning_rate=LinearSchedule(initial_learning_rate, learning_rate_delta, final_learning_rate),
                        gamma=gamma,
                        batch_size=batch_size,
                        experience_buffer_size=experience_buffer_size,
                        target_network_sync_freq=target_network_sync_freq,
                        experience_prob_alpha=experience_prob_alpha,
                        experience_beta=LinearSchedule(initial_experience_beta, experience_beta_delta, final_experience_beta),
                        trajectory_steps=trajectory_steps,
                        sigma_init=initial_sigma,
                        n_atoms=n_atoms,
                        min_value=min_value,
                        max_value=max_value,
                        grad_norm_clip=grad_norm_clip,
                        architecture=dueling_arch,
                        callbacks=CheckpointCallback("experiments/rainbow/", "rainbow", checkpoint_freq),
                        logger=Logger("experiments/rainbow/log_file"),
                        log_freq=10,
                        device=device
                        )
    # rainbowAgent = RainbowAgent.load('experiments/rainbow/rainbow_5000_steps.zip', logger=Logger("experiments/rainbow/log_file_new"))
    rainbowAgent.train(env, num_timesteps)
    generate_graph(rainbowAgent.logger.directory + "scores", rainbowAgent.data['scores'])
    del env
    del rainbowAgent
    