import torch as th

from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger

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
                    grad_norm_clip=1,
                    architecture=None,
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

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_actions = self.model.q_values(samples.next_states).argmax(dim=1)
            next_states_values = self.model.q_target(samples.next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()
