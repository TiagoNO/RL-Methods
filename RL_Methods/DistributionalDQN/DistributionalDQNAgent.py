from matplotlib import projections
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DistributionalDQN.DistributionalModel import DistributionalModel
import torch as th
import numpy as np

class DistributionalDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate,
                    epsilon,
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
                    architecture,
                    device='cpu'
                ):
        self.n_atoms = n_atoms
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(
                        input_dim=input_dim, 
                        action_dim=action_dim, 
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma, 
                        batch_size=batch_size, 
                        experience_buffer_size=experience_buffer_size, 
                        target_network_sync_freq=target_network_sync_freq, 
                        checkpoint_freq=checkpoint_freq, 
                        savedir=savedir, 
                        log_freq=log_freq, 
                        architecture=architecture, 
                        device=device
                        )

    def create_model(self, learning_rate, architecture, device):
        return DistributionalModel(self.input_dim, self.action_dim, learning_rate, self.n_atoms, self.min_value, self.max_value, architecture, device)

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
            projection = self.project_operator(next_best_distrib, samples.rewards, samples.dones)

        loss_v = (-state_log_prob * projection)
        return loss_v.sum(dim=1).mean()

    def project_operator(self, distrib, rewards, dones):
        batch_size = len(rewards)
        projection = th.zeros((batch_size, self.model.n_atoms), dtype=th.float32).to(self.model.device)
        for j in range(self.model.n_atoms):
            atom = self.model.min_v + (j * self.model.delta)
            tz_j = th.clip(rewards + ((~dones) * self.gamma * atom), self.model.min_v, self.model.max_v)
            b_j = (tz_j - self.model.min_v) / self.model.delta
            l = th.floor(b_j).long()
            u = th.ceil(b_j).long()
            eq_mask = u == l
            projection[eq_mask, l[eq_mask]] += distrib[eq_mask, j]
            ne_mask = u != l
            projection[ne_mask, l[ne_mask]] += distrib[ne_mask, j] * (u - b_j)[ne_mask]
            projection[ne_mask, u[ne_mask]] += distrib[ne_mask, j] * (b_j - l)[ne_mask]
        return projection

    @th.no_grad()
    def getAction(self, state, mask=None, deterministic=False):
        self.model.train(True)
        if mask is None:
            mask = np.ones(self.action_dim, dtype=np.bool)

        if np.random.rand() < self.epsilon.get() and not deterministic:
            prob = np.array(mask, dtype=np.float)
            prob /= np.sum(prob)
            random_action = np.random.choice(self.action_dim, 1, p=prob).item()
            return random_action
        else:
            with th.no_grad():
                mask = np.invert(mask)
                state = th.tensor(state, dtype=th.float).to(self.model.device).unsqueeze(0)
                q_values, _ = self.model.q_values(state)
                q_values = q_values.squeeze(0)
                q_values[mask] = -th.inf
                return q_values.argmax().item()
