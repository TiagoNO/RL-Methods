import torch as th
import numpy as np

from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DQN.Distributional.DistributionalModel import DistributionalModel

from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger


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
                    architecture,
                    grad_norm_clip=1,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    log_freq: int = 1,
                    save_log_every=100,
                    device='cpu'
                ):
        super().__init__(
                        input_dim=input_dim, 
                        action_dim=action_dim, 
                        learning_rate=learning_rate,
                        epsilon=epsilon,
                        gamma=gamma, 
                        batch_size=batch_size, 
                        experience_buffer_size=experience_buffer_size, 
                        target_network_sync_freq=target_network_sync_freq, 
                        grad_norm_clip=grad_norm_clip,
                        architecture=architecture,
                        callbacks=callbacks,
                        logger=logger,
                        log_freq=log_freq,
                        save_log_every=save_log_every,
                        device=device
                        )
        self.parameters['n_atoms'] = n_atoms
        self.parameters['min_value'] = min_value
        self.parameters['max_value'] = max_value
        self.model = DistributionalModel(input_dim, action_dim, learning_rate, n_atoms, min_value, max_value, architecture, device)

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.parameters['batch_size'])

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
            tz_j = th.clip(rewards + ((~dones) * self.parameters['gamma'] * atom), self.model.min_v, self.model.max_v)
            b_j = (tz_j - self.model.min_v) / self.model.delta
            l = th.floor(b_j).long()
            u = th.ceil(b_j).long()
            eq_mask = u == l
            projection[eq_mask, l[eq_mask]] += distrib[eq_mask, j]
            ne_mask = u != l
            projection[ne_mask, l[ne_mask]] += distrib[ne_mask, j] * (u - b_j)[ne_mask]
            projection[ne_mask, u[ne_mask]] += distrib[ne_mask, j] * (b_j - l)[ne_mask]
        return projection