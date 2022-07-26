from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DuelingDQN.DuelingModel import DuelingModel


class DuelingDQNAgent(DQNAgent):

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

    def create_model(self, learning_rate, architecture, device):
        return DuelingModel(self.input_dim, self.action_dim, learning_rate, architecture, device)