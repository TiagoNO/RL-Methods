from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DQN.Noisy.NoisyModel import NoisyModel
from RL_Methods.utils.Schedule import LinearSchedule

from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger

class NoisyNetDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate,
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    sigma_init,
                    grad_norm_clip=1,
                    architecture=None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    log_freq: int = 1,
                    device='cpu'
                ):
                
        self.sigma_init = sigma_init
        # Using Noisy network, so we dont need e-greedy search
        # but, for cartpole, initial small epsilon helps convergence
        super().__init__(
                        input_dim=input_dim, 
                        action_dim=action_dim, 
                        learning_rate=learning_rate,
                        epsilon=LinearSchedule(0.1, -1e-4, 0.0),
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

        if not self.logger is None:
            self.logger.log("parameters/sigma_init", self.sigma_init)

    def create_model(self, learning_rate, architecture, device) -> NoisyModel:
        return NoisyModel(self.input_dim, self.action_dim, learning_rate, self.sigma_init, architecture, device)
