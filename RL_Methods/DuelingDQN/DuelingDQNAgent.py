from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DuelingDQN.DuelingModel import DuelingModel
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger

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
                    grad_norm_clip=1,
                    architecture=None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    log_freq: int = 1,
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
                        device=device
                        )

    def create_model(self, learning_rate, architecture, device):
        return DuelingModel(self.input_dim, self.action_dim, learning_rate, architecture, device)