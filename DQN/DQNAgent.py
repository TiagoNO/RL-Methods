from random import random
import numpy as np
import matplotlib.pyplot as plt

from Agent import Agent
import torch as th

from DQN.Model import Model
from Buffers.ExperienceBuffer import ExperienceBuffer

class DQNAgent (Agent):

    def __init__(self, input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay,
                    learning_rate, gamma, batch_size, experience_buffer_size,
                    target_network_sync_freq, device='cpu'):

        self.model = Model(input_dim, action_dim, learning_rate, device)
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.exp_buffer = ExperienceBuffer(experience_buffer_size, self.input_dim, device)
        self.num_timesteps = 0
        self.losses = []
        self.target_network_sync_freq = target_network_sync_freq

    @th.no_grad()
    def getAction(self, state, mask=None, deterministic=False):
        self.model.train(True)
        if mask is None:
            mask = np.ones(self.action_dim, dtype=np.bool)

        if np.random.rand() < self.epsilon and not deterministic:
            prob = np.array(mask, dtype=np.float)
            prob /= np.sum(prob)
            random_action = np.random.choice(self.action_dim, 1, p=prob).item()
            return random_action
        else:
            with th.no_grad():
                mask = np.invert(mask)
                state = th.tensor(state, dtype=th.float).to(self.model.device)
                q_values = self.model.q_values(state)
                q_values[mask] = -th.inf
                return q_values.argmax().item()

    def updateEpsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def step(self):
        self.model.train(True)
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        loss = self.model.loss_func(states_action_values, expected_state_action_values)
        self.model.optimizer.zero_grad()
        self.losses.append(loss.item())
        loss.backward()

        # total_norm = 0
        # for p in self.model.q_net.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)


        th.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.model.optimizer.step()

    def update(self, state, action, reward, done, next_state, info):
        self.exp_buffer.add(state, action, reward, done, next_state)
        self.updateEpsilon()
        self.step()
        self.num_timesteps += 1

        if self.num_timesteps % self.target_network_sync_freq == 0:
            self.sync()


    def endEpisode(self):
        print("     - Epsilon: {}".format(self.epsilon))
        print("     - Steps until sync: {}".format(self.target_network_sync_freq - (self.num_timesteps % self.target_network_sync_freq)))

    def sync(self):
        print("Sync target network...")
        self.model.target_net.load_state_dict(self.model.q_net.state_dict())
                
    def save(self, file):
        th.save(self.model.q_net.state_dict(), file)
        
    def load(self, file):
        self.model.q_net.load_state_dict(th.load(file))
        self.sync() 