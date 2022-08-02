from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DQN.Rainbow.RainbowModel import RainbowModel
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th
import numpy as np

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
                    device='cpu'
                    ):
        self.sigma_init = sigma_init
        self.n_atoms = n_atoms
        self.min_value = min_value
        self.max_value = max_value

        # Using Noisy network, so we dont need e-greedy search
        # but, for cartpole, initial small epsilon helps convergence
        super().__init__(
                        input_dim=input_dim, 
                        action_dim=action_dim, 
                        learning_rate=learning_rate,
                        epsilon=LinearSchedule(0.0, -1e-4, 0.0),
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
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta
        self.trajectory_steps = trajectory_steps
        self.trajectory = []

        if not self.logger is None:
            self.logger.log("parameters/n_atoms", self.n_atoms)
            self.logger.log("parameters/min_value", self.min_value)
            self.logger.log("parameters/max_value", self.max_value)
            self.logger.log("parameters/trajectory_steps", self.trajectory_steps)
            self.logger.log("parameters/sigma_init", self.sigma_init)
            self.logger.log("parameters/experience_prob_alpha", experience_prob_alpha)
            self.logger.log("parameters/experience_beta_initial", self.beta.initial_value)
            self.logger.log("parameters/experience_beta_final", self.beta.final_value)
            self.logger.log("parameters/experience_beta_delta", self.beta.delta)

    def create_model(self, learning_rate, architecture, device):
        return RainbowModel(self.input_dim, self.action_dim, learning_rate, self.sigma_init, self.n_atoms, self.min_value, self.max_value, architecture, device)

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size, self.beta.get())

        _, q_values_atoms = self.model.q_values(samples.states)
        state_action_values = q_values_atoms[range(samples.size), samples.actions]
        state_log_prob = th.log_softmax(state_action_values, dim=1)

        with th.no_grad():
            q_values, _ = self.model.q_values(samples.next_states)
            next_actions = th.argmax(q_values, dim=1)
            _, next_q_atoms = self.model.q_target(samples.next_states)
            next_distrib = th.softmax(next_q_atoms, dim=2)
            next_best_distrib = next_distrib[range(samples.size), next_actions]
            # print(next_best_distrib)
            projection = self.project_operator(next_best_distrib, samples.rewards, samples.dones)

        loss_v = (-state_log_prob * projection).sum(dim=1)

        loss_v *= samples.weights
        self.exp_buffer.update_priorities(samples.indices, loss_v.detach().cpu().numpy())
        self.beta.update()
        self.logger.log("parameters/experience_beta", self.beta.get())
        return loss_v.mean()

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
            reward = (reward * self.gamma) + i[2]
        
        return state, action, reward, done, next_state

    def update(self, state, action, reward, done, next_state, info):
        if len(self.trajectory) >= self.trajectory_steps:
            t_state, t_action, t_reward, t_done, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_done, t_next_state)
            self.step()
            self.trajectory.pop(0)

        self.epsilon.update()
        self.model.update_learning_rate()
        self.trajectory.append([state, action, reward, done, next_state])

        if not self.logger is None and self.num_timesteps % self.log_freq == 0:
            self.logger.log("parameters/learning_rate", self.model.learning_rate.get())
            self.logger.log("parameters/epsilon", self.epsilon.get())

        if self.num_timesteps % self.target_network_sync_freq == 0:
            self.model.sync()

    def print(self):
        super().print()
        print("| Beta: {}\t|".format(self.beta.get()).expandtabs(45))
        print("|" + "=" * 44 + "|")

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
                print(q_values)
                return q_values.argmax().item()

    def loadParameters(self):
        if not self.logger.load():
            return

        super().loadParameters()

        self.experience_prob_alpha = self.logger.data['parameters']['experience_prob_alpha']['data'][-1]
        self.beta.cur_value = self.logger.data['parameters']['experience_beta']['data'][-1]
        self.beta.final_value = self.logger.data['parameters']['experience_beta_final']['data'][-1]
        self.beta.delta = self.logger.data['parameters']['experience_beta_delta']['data'][-1]

        self.trajectory_steps = self.logger.data['parameters']['trajectory_steps']['data'][-1]
        self.sigma_init = self.logger.data['parameters']['sigma_init']['data'][-1]
        self.n_atoms = self.logger.data['parameters']['n_atoms']['data'][-1]
        self.min_value = self.logger.data['parameters']['min_value']['data'][-1]
        self.max_value = self.logger.data['parameters']['max_value']['data'][-1]