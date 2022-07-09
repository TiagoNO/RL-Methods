from DQN.DQNAgent import DQNAgent
from DuelingDQN.DuelingModel import DuelingModel
from Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th


class RainbowAgent(DQNAgent):

    def __init__(self, input_dim, 
                       action_dim, 
                       initial_epsilon, 
                       final_epsilon, 
                       epsilon_decay, 
                       learning_rate, 
                       gamma, 
                       batch_size, 
                       experience_buffer_size, 
                       target_network_sync_freq,
                       experience_prob_alpha, 
                       experience_beta, 
                       experience_beta_decay,
                       device='cpu'):

        super().__init__(input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, device)
        self.model = DuelingModel(input_dim, action_dim, learning_rate, device)
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta
        self.beta_decay = experience_beta_decay

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_actions = self.model.q_values(samples.next_states).argmax(dim=1)
            next_states_values = self.model.q_target(samples.next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()
