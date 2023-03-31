import torch as th
import numpy as np

from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DQN.Rainbow.RainbowModel import RainbowModel
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from RL_Methods.DQN.Noisy.NoisyLinear import NoisyLinear, NoisyFactorizedLinear

from RL_Methods.utils.Logger import Logger, LogLevel
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import LinearSchedule, Schedule

class RainbowAgent(DQNAgent):

    def __init__(self, 
                    input_dim: tuple, 
                    action_dim: int, 
                    learning_rate : Schedule,
                    gamma: float, 
                    batch_size: int, 
                    experience_buffer_size: int, 
                    target_network_sync_freq: int,
                    experience_prob_alpha: float, 
                    experience_beta: Schedule, 
                    trajectory_steps: int,
                    sigma_init: float,
                    n_atoms: int,
                    min_value: float,
                    max_value: float,
                    grad_norm_clip: float = 1,
                    architecture: dict = None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    save_log_every: int = 100,
                    device: str = 'cpu',
                    epsilon: Schedule = None,
                    verbose: LogLevel = LogLevel.INFO
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
                        save_log_every=save_log_every,
                        device=device,
                        verbose=verbose
                        )

        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.model = RainbowModel(input_dim, action_dim, learning_rate, sigma_init, n_atoms, min_value, max_value, architecture, device)

        self.data['parameters']['sigma_init'] = sigma_init
        self.data['parameters']['n_atoms'] = n_atoms
        self.data['parameters']['min_value'] = min_value
        self.data['parameters']['max_value'] = max_value

        self.data['parameters']['experience_beta'] = experience_beta
        self.data['parameters']['experience_prob_alpha'] = experience_prob_alpha

        self.trajectory = []
        self.data['parameters']['trajectory_steps'] = trajectory_steps
        self.data['parameters']['architecture'] = architecture



    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.data['parameters']['batch_size'], self.data['parameters']['experience_beta'].get())
        dones = th.bitwise_or(samples.terminated, samples.truncated)

        _, q_values_atoms = self.model.q_values(samples.states)
        state_action_values = q_values_atoms[range(samples.size), samples.actions]
        state_log_prob = th.log_softmax(state_action_values, dim=1)

        with th.no_grad():
            q_values, _ = self.model.q_values(samples.next_states)
            next_actions = th.argmax(q_values, dim=1)
            _, next_q_atoms = self.model.q_target(samples.next_states)
            next_distrib = th.softmax(next_q_atoms, dim=2)
            next_best_distrib = next_distrib[range(samples.size), next_actions]

            projection = self.project_operator(next_best_distrib, samples.rewards, dones)

        loss_v = (-state_log_prob * projection).sum(dim=1)
        loss_v *= samples.weights

        self.exp_buffer.update_priorities(samples.indices, loss_v.detach().cpu().numpy())
        self.data['parameters']['experience_beta'].update()
        return loss_v.mean()

    def project_operator(self, distrib, rewards, dones):
        batch_size = len(rewards)
        projection = th.zeros((batch_size, self.model.n_atoms), device=self.model.device, dtype=th.float32)

        atoms = (~dones.unsqueeze(1) * (self.data['parameters']['gamma']**self.data['parameters']['trajectory_steps']) * self.model.support_vector.unsqueeze(0))
        tz = th.clip(rewards.unsqueeze(1) + atoms, self.model.min_v, self.model.max_v)
        b = (tz - self.model.min_v) / self.model.delta
        low = th.floor(b).long()
        upper = th.ceil(b).long()

        low[(upper > 0) * (low == upper)] -= 1
        upper[(low < (self.model.n_atoms - 1)) * (low == upper)] += 1

        offset = th.linspace(0, ((batch_size - 1) * self.model.n_atoms), batch_size, device=self.model.device).unsqueeze(1).expand(batch_size, self.model.n_atoms)
        projection.view(-1).index_add_(0, (low + offset).view(-1).long(), (distrib * (upper.float() - b)).view(-1))
        projection.view(-1).index_add_(0, (upper + offset).view(-1).long(), (distrib * (b - low.float())).view(-1))

        # freeing the cuda memory
        del low
        del upper
        del b
        del tz
        del atoms
        del offset

        return projection


    def beginEpisode(self):
        while(len(self.trajectory) > 0):
            state, action, reward, terminated, truncated, next_state = self.getTrajectory()
            self.exp_buffer.add(state, action, reward, terminated, truncated, next_state)
            self.trajectory.pop(0)

    def getTrajectory(self):
        if len(self.trajectory) == 1:
            return self.trajectory[0]

        state = self.trajectory[0][0]
        action = self.trajectory[0][1]
        reward = 0
        terminated = self.trajectory[0][3]
        truncated = self.trajectory[0][4]
        next_state = self.trajectory[-1][5]

        for i in reversed(self.trajectory):
            reward = (reward * self.data['parameters']['gamma']) + i[2]
        
        return state, action, reward, terminated, truncated, next_state

    def update(self, state, action, reward, terminated, truncated, next_state, info):
        if len(self.trajectory) >= self.data['parameters']['trajectory_steps']:
            t_state, t_action, t_reward, t_terminated, t_truncated, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_terminated, t_truncated, t_next_state)
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, terminated, truncated, next_state])

    def endEpisode(self):
        if self.data['parameters']['verbose'] >= 1:
            self.log(LogLevel.DEBUG, "parameters/beta", self.data['parameters']['experience_beta'].get())
            for idx, p in enumerate(self.model.features_extractor.modules()): 
                if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                    self.log(LogLevel.DEBUG, "parameters/feature_extractor_L{}_avg_noisy".format(idx), p.sigma_weight.mean().item())

            for idx, p in enumerate(self.model.advantage_net.modules()): 
                if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                    self.log(LogLevel.DEBUG, "parameters/advantage_L{}_avg_noisy".format(idx), p.sigma_weight.mean().item())

            for idx, p in enumerate(self.model.value_net.modules()): 
                if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                    self.log(LogLevel.DEBUG, "parameters/value_L{}_avg_noisy".format(idx), p.sigma_weight.mean().item())
        super().endEpisode()        