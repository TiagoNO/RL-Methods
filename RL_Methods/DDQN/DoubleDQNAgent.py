from RL_Methods.DQN.DQNAgent import DQNAgent

import torch as th

class DoubleDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate, 
                    epsilon,
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    checkpoint_freq,
                    savedir,
                    log_freq,
                    architecture=None,
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
                        checkpoint_freq=checkpoint_freq, 
                        savedir=savedir, 
                        log_freq=log_freq, 
                        architecture=architecture, 
                        device=device
                    )

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_actions = self.model.q_values(samples.next_states).argmax(dim=1)
            next_states_values = self.model.q_target(samples.next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()
