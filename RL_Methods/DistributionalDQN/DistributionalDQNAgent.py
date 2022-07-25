from matplotlib import projections
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DistributionalDQN.DistributionalModel import DistributionalModel
import torch as th
import numpy as np

class DistributionalDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    initial_epsilon, 
                    final_epsilon, 
                    epsilon_decay, 
                    learning_rate, 
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    n_atoms,
                    min_value,
                    max_value,
                    checkpoint_freq,
                    savedir,
                    log_freq,
                    device='cpu'
                ):
                
        super().__init__(input_dim, action_dim, initial_epsilon, final_epsilon, 
                        epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, 
                        target_network_sync_freq, checkpoint_freq, savedir, log_freq, device)
        self.model = DistributionalModel(input_dim, action_dim, learning_rate, n_atoms, min_value, max_value, device)

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        # calculating q_values distribution
        _, q_atoms = self.model.q_values(samples.states)
        states_action_values = q_atoms[range(samples.size), samples.actions]
        state_log_prob = th.log_softmax(states_action_values, dim=1)

        # using no grad to avoid updating the target network
        with th.no_grad():
            next_q_values, next_atoms = self.model.q_target(samples.next_states)
            next_actions = th.argmax(next_q_values, dim=1)
            next_distrib = th.softmax(next_atoms, dim=2)
            next_best_distrib = next_distrib[range(samples.size), next_actions]
            projection = self.project_operator(next_best_distrib.numpy(), samples.rewards.numpy(), samples.dones.numpy())

        loss_v = (-state_log_prob * projection)
        return loss_v.sum(dim=1).mean()

    def project_operator(self, distrib, rewards, dones):
        batch_size = len(rewards)
        projection = np.zeros((batch_size, self.model.n_atoms), dtype=np.float32)
        for j in range(self.model.n_atoms):
            atom = self.model.min_v + (j * self.model.delta)
            tz_j = np.clip(rewards + ((1 - dones) * self.gamma * atom), self.model.min_v, self.model.max_v)
            b_j = (tz_j - self.model.min_v) / self.model.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            projection[eq_mask, l[eq_mask]] += distrib[eq_mask, j]
            ne_mask = u != l
            projection[ne_mask, l[ne_mask]] += distrib[ne_mask, j] * (u - b_j)[ne_mask]
            projection[ne_mask, u[ne_mask]] += distrib[ne_mask, j] * (b_j - l)[ne_mask]
        return th.tensor(projection, dtype=th.float32)

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
                state = th.tensor(state, dtype=th.float).to(self.model.device).unsqueeze(0)
                q_values, _ = self.model.q_values(state)
                # q_values[mask] = -th.inf
                return q_values.argmax().item()