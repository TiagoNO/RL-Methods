from NNMethods.DQN.DQNAgent import DQNAgent

import torch as th

class DoubleDQNAgent(DQNAgent):

    def __init__(self, input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, device='cpu'):
        super().__init__(input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, device)

    def step(self):
        self.model.train(True)
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_actions = self.model.q_values(samples.next_states).argmax(dim=1)
            next_states_values = self.model.q_target(samples.next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        loss = self.model.loss_func(states_action_values, expected_state_action_values)
        self.model.optimizer.zero_grad()
        self.losses.append(loss.item())
        loss.backward()

        # total_norm = 0
        # for p in self.model.q_net.parameters():
        #     param_norm = p.grad.data.norm(2)
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)

        
        th.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.model.optimizer.step()

#next_actions = self.model.q_values(samples.next_states).argmax(dim=1)
#next_states_values = self.model.q_target(samples.next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
