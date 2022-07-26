from matplotlib import projections
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.Rainbow.RainbowModel import RainbowModel
from RL_Methods.DuelingDQN.DuelingModel import DuelingModel
from RL_Methods.NoisyNetDQN.NoisyModel import NoisyModel
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th
import numpy as np

from RL_Methods.utils.Schedule import LinearSchedule, Schedule


class RainbowAgent(DQNAgent):

    def __init__(self, input_dim, 
                       action_dim, 
                       learning_rate : Schedule,
                       epsilon : Schedule,
                       gamma, 
                       batch_size, 
                       experience_buffer_size, 
                       target_network_sync_freq,
                       experience_prob_alpha, 
                       experience_beta : Schedule, 
                       trajectory_steps,
                       initial_sigma,
                       n_atoms,
                       min_value,
                       max_value,
                       checkpoint_freq,
                       savedir,
                       log_freq,
                       architecture,
                       device='cpu'):
        self.initial_sigma = initial_sigma
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
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta
        self.trajectory_steps = trajectory_steps
        self.trajectory = []

        # Using Noisy network, so we dont need e-greedy search
        # but, for cartpole, initial small epsilon helps convergence
        self.epsilon = LinearSchedule(0.1, epsilon.delta, 0.0)

    def create_model(self, learning_rate, architecture, device):
        return RainbowModel(self.input_dim, self.action_dim, learning_rate, self.initial_sigma, self.n_atoms, self.min_value, self.max_value, architecture, device)

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
            projection = self.project_operator(next_best_distrib.numpy(), samples.rewards.numpy(), samples.dones.numpy())

        loss_v = (-state_log_prob * projection).sum(dim=1)

        loss_v *= samples.weights
        self.exp_buffer.update_priorities(samples.indices, loss_v.detach().cpu().numpy())
        self.beta.update()
        return loss_v.mean()

    def project_operator(self, distrib, rewards, dones):
        batch_size = len(rewards)
        projection = np.zeros((batch_size, self.model.n_atoms), dtype=np.float32)
        for j in range(self.model.n_atoms):
            atom = self.model.min_v + (j * self.model.delta)
            tz_j = np.clip(rewards + ((1 - dones) * (self.gamma**self.trajectory_steps) * atom), self.model.min_v, self.model.max_v)
            b_j = (tz_j - self.model.min_v) / self.model.delta
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            projection[eq_mask, l[eq_mask]] += distrib[eq_mask, j]
            ne_mask = u != l
            projection[ne_mask, l[ne_mask]] += distrib[ne_mask, j] * (u - b_j)[ne_mask]
            projection[ne_mask, u[ne_mask]] += distrib[ne_mask, j] * (b_j - l)[ne_mask]
        return th.tensor(projection, dtype=th.float32)

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
            self.epsilon.update()
            self.model.update_learning_rate()
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, done, next_state])

        if self.num_timesteps % self.checkpoint_freq  == 0:
            self.save(self.savedir + "dqn_{}_steps.pt".format(self.num_timesteps))

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
                return q_values.argmax().item()