from DQN.DQNAgent import DQNAgent
from Rainbow.RainbowModel import RainbowModel
from DuelingDQN.DuelingModel import DuelingModel
from NoisyNetDQN.NoisyModel import NoisyModel
from Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th
import numpy as np


class RainbowAgent(DQNAgent):

    def __init__(self, input_dim, 
                       action_dim, 
                       initial_epsilon, 
                       final_epsilon, 
                       epsilon_decay, 
                       learning_rate, 
                       gamma, 
                       batch_size, 
                       experience_buffer_size, 
                       target_network_sync_freq,
                       experience_prob_alpha, 
                       experience_beta, 
                       experience_beta_decay,
                       trajectory_steps,
                       initial_sigma,
                       device='cpu'):

        super().__init__(input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, device)
        self.model = RainbowModel(input_dim, action_dim, learning_rate, initial_sigma, device)
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta
        self.beta_decay = experience_beta_decay
        self.trajectory_steps = trajectory_steps
        self.trajectory = []

        # no need to exploration with e-greedy, using Noisy network
        self.epsilon = 0
        self.final_epsilon = 0

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_actions = self.model.q_values(samples.next_states).argmax(dim=1)
            next_states_values = self.model.q_target(samples.next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def beginEpisode(self, state):
        for i in range(len(self.trajectory)):
            state, action, reward, done, next_state = self.getTrajectory()
            self.exp_buffer.add(state, action, reward, done, next_state)
            self.trajectory.pop(0)

    def getTrajectory(self):
        state = self.trajectory[0][0]
        action = self.trajectory[0][1]
        reward = 0
        done = self.trajectory[0][3]
        next_state = self.trajectory[-1][4]

        for i in reversed(self.trajectory):
            reward = (reward * self.gamma) + i[2]
        
        return state, action, reward, done, next_state

    def update(self, state, action, reward, done, next_state, info):
        if len(self.trajectory) >= self.trajectory_steps:
            t_state, t_action, t_reward, t_done, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_done, t_next_state)
            self.updateEpsilon()
            self.step()
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, done, next_state])

        if self.num_timesteps % self.target_network_sync_freq == 0:
            self.model.sync()

    @th.no_grad()
    def getAction(self, state, mask=None, deterministic=False):
        self.model.train(True)
        if mask is None:
            mask = np.ones(self.action_dim, dtype=np.bool)

        with th.no_grad():
            mask = np.invert(mask)
            state = th.tensor(state, dtype=th.float).to(self.model.device)
            q_values = self.model.q_values(state)
            q_values[mask] = -th.inf
            return q_values.argmax().item()