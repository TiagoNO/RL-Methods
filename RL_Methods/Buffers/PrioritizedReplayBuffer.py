import torch as th
import numpy as np
from RL_Methods.Buffers.ExperienceBuffer import *

class PrioritizedExperienceSamples(ExperienceSamples):
    def __init__(self, states, actions, rewards, terminated, truncated, next_states, indices, weights, device):
        super().__init__(states, actions, rewards, terminated, truncated, next_states, indices, device)
        self.weights = th.from_numpy(weights).to(device)

class PrioritizedReplayBuffer(ExperienceBuffer):

    def __init__(self, max_sz, input_dim, device, prob_alpha=0.6):
        super().__init__(max_sz, input_dim, device)
        self.prob_alpha = prob_alpha
        self.priorities = np.full((self.max_sz, ), fill_value=1e-5, dtype=np.float32)
        self.max_priority = 1e-5
        self.min_priority = 1

    def add(self, state, action, reward, terminated, truncated, next_state):
        self.priorities[self.pos] = self.max_priority
        super().add(state, action, reward, terminated, truncated, next_state)

    def sample(self, batch_size, beta=0.4):
        if self.curr_sz < self.max_sz:
            probs = self.priorities[0:self.pos].copy()
        else:
            probs = self.priorities.copy()

        probs /= probs.sum()
        sample_sz = min(self.curr_sz, batch_size)
        indices = np.random.choice(self.curr_sz, sample_sz, p=probs, replace=True)
        weights = (self.curr_sz * probs[indices]) ** (-beta)
        weights /= max(1e-5, self.min_priority)
        return PrioritizedExperienceSamples(
                    self.obs[indices], 
                    self.actions[indices], 
                    self.rewards[indices],
                    self.terminated[indices], 
                    self.truncated[indices], 
                    self.obs_[indices], 
                    indices,
                    weights,
                    self.device
                    )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            prio_alpha = (prio**self.prob_alpha) + 1e-5
            self.priorities[idx] = prio_alpha
            self.max_priority = max(prio_alpha, self.max_priority)
            self.min_priority = min(prio_alpha, self.min_priority)
