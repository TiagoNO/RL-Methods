from DQN.DQNAgent import DQNAgent
from Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th

class PrioritizedDQN(DQNAgent):

    def __init__(self, input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, experience_prob_alpha, experience_beta, experience_beta_decay, device='cpu'):
        super().__init__(input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, device)
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta
        self.beta_decay = experience_beta_decay

    def updateBeta(self):
        self.beta = min(1, self.beta + self.beta_decay)

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        loss = self.model.loss_func(states_action_values, expected_state_action_values)
        loss *= samples.weights
        self.exp_buffer.update_priorities(samples.indices, loss.detach().cpu().numpy())
        self.updateBeta()
        return loss.mean()


    def print(self):
        super().print()
        print("|     - Beta: {}\t|".format(self.beta).expandtabs(45))
        print("|" + "=" * 44 + "|")

    def endEpisode(self):
        super().endEpisode()
