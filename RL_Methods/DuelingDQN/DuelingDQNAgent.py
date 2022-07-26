from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DuelingDQN.DuelingModel import DuelingModel


class DuelingDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    initial_epsilon, 
                    final_epsilon, 
                    epsilon_decay, 
                    learning_rate, 
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
                
        super().__init__(input_dim, action_dim, initial_epsilon, 
                        final_epsilon, epsilon_decay, learning_rate, 
                        gamma, batch_size, experience_buffer_size, target_network_sync_freq, 
                        checkpoint_freq, savedir, log_freq, architecture, device)
        self.model = DuelingModel(input_dim, action_dim, learning_rate, architecture, device)
        # print(self.model)
