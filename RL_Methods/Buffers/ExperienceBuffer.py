import torch as th
import numpy as np

class ExperienceSamples:

    def __init__(self, states, actions, rewards, dones, next_states, indices, device):
        self.states = th.from_numpy(states).to(device)
        self.actions = th.from_numpy(actions).to(device)
        self.rewards = th.from_numpy(rewards).to(device)
        self.dones = th.from_numpy(dones).to(device)
        self.next_states = th.from_numpy(next_states).to(device)
        self.indices = th.from_numpy(indices).to(device)
        self.size = len(rewards)

class ExperienceBuffer:
    def __init__(self, max_sz, input_dim, device):
        self.max_sz = int(max_sz)
        self.obs = np.zeros((self.max_sz, input_dim[0]), dtype=np.float32)
        self.actions = np.zeros(self.max_sz, dtype=np.int64)
        self.rewards = np.zeros(self.max_sz, dtype=np.float32)
        self.dones = np.zeros(self.max_sz, dtype=bool)
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
        return ExperienceSamples(self.obs[indices], self.actions[indices], self.rewards[indices], self.dones[indices], self.obs_[indices], indices, self.device)

    def __len__(self):
        return self.curr_sz