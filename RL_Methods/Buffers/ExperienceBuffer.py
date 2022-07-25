import torch as th
import numpy as np

class ExperienceSamples:

    def __init__(self, states, actions, rewards, dones, next_states, device):
        self.states = th.tensor(np.array(states, copy=False), dtype=th.float).to(device)
        self.actions = th.tensor(np.array(actions, copy=False), dtype=th.int64).to(device)
        self.rewards = th.tensor(np.array(rewards, copy=False), dtype=th.float32).to(device)
        self.dones = th.tensor(np.array(dones, copy=False), dtype=th.bool).to(device)
        self.next_states = th.tensor(np.array(next_states, copy=False), dtype=th.float).to(device)
        self.size = len(rewards)

class ExperienceBuffer:
    def __init__(self, max_sz, input_dim, device):
        self.max_sz = int(max_sz)
        self.obs = np.zeros((self.max_sz, input_dim[0]), dtype=np.float32)
        self.actions = np.zeros(self.max_sz, dtype=np.int64)
        self.rewards = np.zeros(self.max_sz, dtype=np.float32)
        self.dones = np.zeros(self.max_sz, dtype=np.bool_)
        self.obs_ = np.zeros((self.max_sz, input_dim[0]), dtype=np.float32)
        self.pos = 0
        self.curr_sz = 0
        self.device = device

    def add(self, state, action, reward, done, next_state):
        self.obs[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.obs_[self.pos] = next_state
        self.pos = (self.pos + 1) % self.max_sz
        self.curr_sz = min(self.max_sz, self.curr_sz + 1)

    def sample(self, batch_sz):
        sample_sz = min(self.curr_sz, batch_sz)
        indices = np.random.choice(self.curr_sz, sample_sz, replace=False)
        return ExperienceSamples(self.obs[indices], self.actions[indices], self.rewards[indices], self.dones[indices], self.obs_[indices], self.device)

    def __len__(self):
        return self.curr_sz