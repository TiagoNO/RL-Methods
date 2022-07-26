from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.NoisyNetDQN.NoisyModel import NoisyModel
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
        self.model = NoisyModel(input_dim, action_dim, learning_rate, sigma_init, architecture, device)
        self.epsilon = epsilon

    @th.no_grad()
    def getAction(self, state, mask=None, deterministic=False):
        self.model.train(True)
        if mask is None:
            mask = np.ones(self.action_dim, dtype=np.bool)

        with th.no_grad():
            mask = np.invert(mask)
            state = th.tensor(state, dtype=th.float).to(self.model.device)
            q_values = self.model.q_values(state)
            q_values[mask] = -th.inf
            return q_values.argmax().item()