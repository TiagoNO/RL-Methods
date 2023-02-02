import torch as th
import numpy as np

class ExperienceSamples:

    def __init__(self, states, actions, rewards, terminated, truncated, next_states, indices, device):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.terminated = terminated
        self.truncated = truncated
        self.next_states = next_states
        self.indices = indices
        self.size = rewards.shape[0]

class ExperienceBuffer:
    def __init__(self, max_sz, input_dim, device):
        self.max_sz = int(max_sz)
        self.obs = th.zeros((self.max_sz, input_dim[0]), dtype=th.float32, device=device)
        self.actions = th.zeros(self.max_sz, dtype=th.int64, device=device)
        self.rewards = th.zeros(self.max_sz, dtype=th.float32, device=device)
        self.terminated = th.zeros(self.max_sz, dtype=th.bool, device=device)
        self.truncated = th.zeros(self.max_sz, dtype=th.bool, device=device)
        self.obs_ = th.zeros((self.max_sz, input_dim[0]), dtype=th.float32, device=device)
        self.pos = 0
        self.curr_sz = 0
        self.device = device

    def add(self, state, action, reward, terminated, truncated, next_state):
        self.obs[self.pos] = th.from_numpy(state).to(self.device)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminated[self.pos] = terminated
        self.truncated[self.pos] = truncated
        self.obs_[self.pos] = th.from_numpy(next_state).to(self.device)
        self.pos = (self.pos + 1) % self.max_sz
        self.curr_sz = min(self.max_sz, self.curr_sz + 1)

    def sample(self, batch_sz):
        sample_sz = min(self.curr_sz, batch_sz)
        indices = th.randint(self.curr_sz, (sample_sz,))
        # indices = np.random.choice(self.curr_sz, sample_sz, replace=False)
        return ExperienceSamples(self.obs[indices], self.actions[indices], self.rewards[indices], self.terminated[indices], self.truncated[indices], self.obs_[indices], indices, self.device)

    def __len__(self):
        return self.curr_sz