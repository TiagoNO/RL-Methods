import torch as th
import numpy as np

from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DQN.Rainbow.RainbowModel import RainbowModel
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer, OptimizedPrioritizedReplayBuffer
from RL_Methods.DQN.Noisy.NoisyLinear import NoisyLinear, NoisyFactorizedLinear

from RL_Methods.utils.Logger import Logger
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import LinearSchedule, Schedule

class RainbowAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate : Schedule,
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    experience_prob_alpha, 
                    experience_beta : Schedule, 
                    trajectory_steps,
                    sigma_init,
                    n_atoms,
                    min_value,
                    max_value,
                    grad_norm_clip=1,
                    architecture=None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    log_freq: int = 1,
                    save_log_every=100,
                    device='cpu',
                    epsilon=None,
                    verbose=0
                    ):

        if epsilon is None:
            epsilon=LinearSchedule(0.0, -1e-4, 0.0)

        # Using Noisy network, so we dont need e-greedy search
        # but, for cartpole, initial small epsilon helps convergence
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
                        architecture=None,
                        callbacks=callbacks,
                        logger=logger,
                        log_freq=log_freq,
                        save_log_every=save_log_every,
                        device=device,
                        verbose=verbose
                        )

        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.model = RainbowModel(input_dim, action_dim, learning_rate, sigma_init, n_atoms, min_value, max_value, architecture, device)

        self.parameters['sigma_init'] = sigma_init
        self.parameters['n_atoms'] = n_atoms
        self.parameters['min_value'] = min_value
        self.parameters['max_value'] = max_value

        self.parameters['experience_beta'] = experience_beta
        self.parameters['experience_prob_alpha'] = experience_prob_alpha

        self.trajectory = []
        self.parameters['trajectory_steps'] = trajectory_steps
        self.parameters['architecture'] = architecture



    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.parameters['batch_size'], self.parameters['experience_beta'].get())

        _, q_values_atoms = self.model.q_values(samples.states)
        state_action_values = q_values_atoms[range(samples.size), samples.actions]
        state_log_prob = th.log_softmax(state_action_values, dim=1)

        with th.no_grad():
            q_values, _ = self.model.q_values(samples.next_states)
            next_actions = th.argmax(q_values, dim=1)
            _, next_q_atoms = self.model.q_target(samples.next_states)
            next_distrib = th.softmax(next_q_atoms, dim=2)
            next_best_distrib = next_distrib[range(samples.size), next_actions]

            projection = self.project_operator(next_best_distrib, samples.rewards, samples.dones)

        loss_v = (-state_log_prob * projection).sum(dim=1)
        loss_v *= samples.weights

        self.exp_buffer.update_priorities(samples.indices, loss_v.detach().cpu().numpy())
        self.parameters['experience_beta'].update()
        return loss_v.mean()

    def project_operator(self, distrib, rewards, dones):
        batch_size = len(rewards)
        projection = th.zeros((batch_size, self.model.n_atoms), device=self.model.device, dtype=th.float32)

        atoms = (~dones.unsqueeze(1) * (self.parameters['gamma']**self.parameters['trajectory_steps']) * self.model.support_vector.unsqueeze(0))
        tz = th.clip(rewards.unsqueeze(1) + atoms, self.model.min_v, self.model.max_v)
        b = (tz - self.model.min_v) / self.model.delta
        low = th.floor(b).long()
        upper = th.ceil(b).long()

        low[(upper > 0) * (low == upper)] -= 1
        upper[(low < (self.model.n_atoms - 1)) * (low == upper)] += 1

        offset = th.linspace(0, ((batch_size - 1) * self.model.n_atoms), batch_size, device=self.model.device).unsqueeze(1).expand(batch_size, self.model.n_atoms)
        projection.view(-1).index_add_(0, (low + offset).view(-1).long(), (distrib * (upper.float() - b)).view(-1))
        projection.view(-1).index_add_(0, (upper + offset).view(-1).long(), (distrib * (b - low.float())).view(-1))
        return projection


    def beginEpisode(self, state):
        for _ in range(len(self.trajectory)):
            state, action, reward, done, next_state = self.getTrajectory()
            self.exp_buffer.add(state, action, reward, done, next_state)
            self.trajectory.pop(0)

    def getTrajectory(self):
        if len(self.trajectory) == 1:
            return self.trajectory[0]

        state = self.trajectory[0][0]
        action = self.trajectory[0][1]
        reward = 0
        done = self.trajectory[0][3]
        next_state = self.trajectory[-1][4]

        for i in reversed(self.trajectory):
            reward = (reward * self.parameters['gamma']) + i[2]
        
        return state, action, reward, done, next_state

    def update(self, state, action, reward, done, next_state, info):
        if len(self.trajectory) >= self.parameters['trajectory_steps']:
            t_state, t_action, t_reward, t_done, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_done, t_next_state)
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, done, next_state])

    def endEpisode(self):
        if not self.logger is None and self.parameters['verbose'] >= 1:
            self.logger.log("parameters/beta", self.parameters['experience_beta'].get())
            for idx, p in enumerate(self.model.features_extractor.modules()): 
                if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                    self.logger.log("parameters/feature_extractor_L{}_avg_noisy".format(idx), p.sigma_weight.mean().item())

            for idx, p in enumerate(self.model.advantage_net.modules()): 
                if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                    self.logger.log("parameters/advantage_L{}_avg_noisy".format(idx), p.sigma_weight.mean().item())

            for idx, p in enumerate(self.model.value_net.modules()): 
                if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                    self.logger.log("parameters/value_L{}_avg_noisy".format(idx), p.sigma_weight.mean().item())
        super().endEpisode()        