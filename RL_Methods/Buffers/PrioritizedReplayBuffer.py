import torch as th
import numpy as np
from RL_Methods.Buffers.ExperienceBuffer import *

class PrioritizedExperienceSamples(ExperienceSamples):
    def __init__(self, states, actions, rewards, dones, next_states, indices, weights, device):
        super().__init__(states, actions, rewards, dones, next_states, device)
        self.indices = th.from_numpy(indices).to(device)
        self.weights = th.from_numpy(weights).to(device)

class PrioritizedReplayBuffer(ExperienceBuffer):

    def __init__(self, max_sz, input_dim, device, prob_alpha=0.6):
        super().__init__(max_sz, input_dim, device)
        self.prob_alpha = prob_alpha
        self.priorities = np.full((self.max_sz, ), fill_value=1e-5, dtype=np.float32)
        self.max_priority = 1e-5
        self.min_priority = 1

    def add(self, state, action, reward, done, next_state):
        self.priorities[self.pos] = self.max_priority
        super().add(state, action, reward, done, next_state)

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
                    self.dones[indices], 
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

class OptimizedPrioritizedReplayBuffer(ExperienceBuffer):

    def __init__(self, max_sz, input_dim, device, prob_alpha=0.6):
        super().__init__(max_sz, input_dim, device)
        self.prob_alpha = prob_alpha
        self.sum_prios = np.zeros(2 * self.max_sz)
        self.min_prios = np.full(2 * self.max_sz, np.inf)
        self.max_priority = 1

    def _set_priority(self, index, priority):
        index += self.max_sz
        self.sum_prios[index] = priority
        index //= 2
        while index >= 1:
            self.sum_prios[index] = self.sum_prios[2 * index] + self.sum_prios[(2 * index) + 1]
            self.min_prios[index] = min(self.min_prios[2 * index], self.min_prios[(2 * index) + 1])
            index //= 2

    def _sum(self):
        return self.sum_prios[1]

    def _min(self):
        return self.min_prios[1]

    def _get_sum_index(self, sum_value):
        index = 1
        while index < self.max_sz:
            if self.sum_prios[index * 2] > sum_value:
                index = 2 * index 
            else:
                sum_value -= self.sum_prios[index * 2]
                index = (2 * index) + 1
        return index - self.max_sz

    def add(self, state, action, reward, done, next_state):
        priority_alpha = (self.max_priority ** self.prob_alpha)
        self._set_priority(self.pos, priority_alpha)
        super().add(state, action, reward, done, next_state)

    def sample(self, batch_size, beta=0.4):
        indices = np.zeros(batch_size, np.int64)
        weights = np.zeros(batch_size, dtype=np.float)

        min_prob = self._min() / max(self._sum(), 1e-5)
        max_weight = (min_prob * self.curr_sz) ** (-beta)

        for i in range(batch_size):
            sum_value = np.random.rand() * self._sum()
            index = self._get_sum_index(sum_value)

            prob = self.sum_prios[index + self.max_sz] / self._sum()
            weight = (prob * self.curr_sz) ** (-beta)

            indices[i] = index
            weights[i] = weight / max(max_weight, 1e-5)

        return PrioritizedExperienceSamples(
                    self.obs[indices], 
                    self.actions[indices], 
                    self.rewards[indices], 
                    self.dones[indices], 
                    self.obs_[indices], 
                    indices,
                    weights,
                    self.device
                    )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.max_priority = max(prio, self.max_priority)
            prio_alpha = (prio ** self.prob_alpha)
            self._set_priority(idx, prio_alpha)