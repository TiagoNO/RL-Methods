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
                    device='cpu',
                    verbose=0
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
                        device=device,
                        verbose=verbose
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
            projection = self.project_operator(next_best_distrib.cpu(), samples.rewards.cpu(), samples.dones.cpu())

        loss_v = (-state_log_prob * projection)
        return loss_v.sum(dim=1).mean()

    def project_operator(self, distrib, rewards, dones):
        batch_size = len(rewards)
        projection = th.zeros((batch_size, self.model.n_atoms), dtype=th.float32)

        atoms = (~dones.unsqueeze(1) * (self.parameters['gamma']**self.parameters['trajectory_steps']) * self.model.support_vector.unsqueeze(0).to('cpu'))
        tz = th.clip(rewards.unsqueeze(1) + atoms, self.model.min_v, self.model.max_v)
        b = (tz - self.model.min_v) / self.model.delta
        low = th.floor(b).long()
        upper = th.ceil(b).long()

        low[(upper > 0) * (low == upper)] -= 1
        upper[(low < (self.model.n_atoms - 1)) * (low == upper)] += 1

        offset = th.linspace(0, ((batch_size - 1) * self.model.n_atoms), batch_size).unsqueeze(1).expand(batch_size, self.model.n_atoms)
        projection.view(-1).index_add_(0, (low + offset).view(-1).long(), (distrib * (upper.float() - b)).view(-1))
        projection.view(-1).index_add_(0, (upper + offset).view(-1).long(), (distrib * (b - low.float())).view(-1))
        return projection.to(self.model.device)