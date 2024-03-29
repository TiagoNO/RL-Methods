from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.DQN.Noisy.NoisyModel import NoisyModel
from RL_Methods.DQN.Noisy.NoisyLinear import NoisyLinear, NoisyFactorizedLinear

from RL_Methods.utils.Schedule import LinearSchedule

from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger, LogLevel

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
                    save_log_every=100,
                    device='cpu',
                    epsilon=None,
                    verbose=0
                ):
        if epsilon is None:
            epsilon=LinearSchedule(0.1, -1e-4, 0.0)

        # Using Noisy network, so we dont need e-greedy search
        # but, for cartpole, initial small epsilon helps convergence
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
                        save_log_every=save_log_every,
                        device=device,
                        verbose=verbose
                        )

        self.data['parameters']['sigma_init'] = sigma_init
        self.model = NoisyModel(input_dim, action_dim, learning_rate, sigma_init, architecture, device)

    def endEpisode(self):
        for idx, p in enumerate(self.model.q_net.modules()): 
            if type(p) == NoisyLinear or type(p) == NoisyFactorizedLinear:
                self.log(LogLevel.DEBUG, "parameters/layer_{}_avg_noisy".format(idx), p.sigma_weight.mean().item())
        super().endEpisode()