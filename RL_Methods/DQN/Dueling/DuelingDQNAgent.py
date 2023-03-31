from RL_Methods.utils.Logger import Logger, LogLevel
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.utils.Callback import Callback
from RL_Methods.DQN.Dueling.DuelingModel import DuelingModel

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
                    save_log_every=100,
                    device='cpu',
                    verbose: LogLevel = LogLevel.INFO
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
                        architecture=None,
                        callbacks=callbacks,
                        logger=logger,
                        save_log_every=save_log_every,
                        device=device,
                        verbose=verbose
                        )

        self.model = DuelingModel(input_dim, action_dim, learning_rate, architecture, device)
        self.data['parameters']['architecture'] = architecture

