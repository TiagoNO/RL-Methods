from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.NoisyNetDQN.NoisyModel import NoisyModel
from RL_Methods.utils.Schedule import LinearSchedule
import torch as th
import numpy as np

class NoisyNetDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate,
                    epsilon,
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    sigma_init,
                    checkpoint_freq,
                    savedir,
                    log_freq,
                    architecture,
                    device='cpu'
                ):
                
        self.sigma_init = sigma_init
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

        # Using Noisy network, so we dont need e-greedy search
        # but, for cartpole, initial small epsilon helps convergence
        self.epsilon = LinearSchedule(0.1, epsilon.delta, 0.0)

    def create_model(self, learning_rate, architecture, device):
        return NoisyModel(self.input_dim, self.action_dim, learning_rate, self.sigma_init, architecture, device)
