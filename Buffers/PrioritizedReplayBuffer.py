import numpy as np
import torch as th

from Buffers.ExperienceBuffer import ExperienceBuffer, ExperienceSamples

class PrioritizedExperienceSamples(ExperienceSamples):
    def __init__(self, states, actions, rewards, dones, next_states, indices, weights, device):
        super().__init__(states, actions, rewards, dones, next_states, device)
        self.indices = th.tensor(np.array(indices, copy=False), dtype=th.int64).to(device)
        self.weights = th.tensor(np.array(weights, copy=False), dtype=th.float).to(device)

class PrioritizedReplayBuffer(ExperienceBuffer):

    def __init__(self, max_sz, input_dim, device, prob_alpha=0.6):
        super().__init__(max_sz, input_dim, device)
        self.prob_alpha = prob_alpha
        self.priorities = np.zeros((self.max_sz, ), dtype=np.float32)

    def add(self, state, action, reward, done, next_state):
        super().add(state, action, reward, done, next_state)

    def sample(self, batch_size, beta=0.4):
        if self.curr_sz == self.max_sz:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = (prios ** self.prob_alpha) + 1e-5

        probs /= probs.sum()
        sample_sz = min(self.curr_sz, batch_size)
        indices = np.random.choice(self.curr_sz, sample_sz, p=probs)
        weights = (self.curr_sz * probs[indices]) ** (-beta)
        weights /= weights.max()

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
            self.priorities[idx] = prio