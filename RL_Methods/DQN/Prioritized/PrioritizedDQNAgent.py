from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th

from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger
import time

class PrioritizedDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim: tuple, 
                    action_dim: int, 
                    learning_rate: Schedule,
                    epsilon: Schedule,
                    gamma: float, 
                    batch_size: int, 
                    experience_buffer_size: int, 
                    target_network_sync_freq: int, 
                    experience_prob_alpha: float, 
                    experience_beta : Schedule, 
                    grad_norm_clip: float = 1,
                    architecture:dict = None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    save_log_every: int = 100,
                    device: str = 'cpu',
                    verbose : int = 0
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
                        save_log_every=save_log_every,
                        device=device,
                        verbose=verbose
                        )
        self.parameters['experience_beta'] = experience_beta
        self.parameters['experience_prob_alpha'] = experience_prob_alpha
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        # self.exp_buffer = OptimizedPrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.sample_time = 0
        self.count = 1

    def calculate_loss(self):
        begin = time.time()
        samples = self.exp_buffer.sample(self.parameters['batch_size'], self.parameters['experience_beta'].get())
        dones = th.bitwise_or(samples.terminated, samples.truncated)

        self.sample_time += time.time() - begin
        self.count += 1

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~dones) * self.parameters['gamma'] * next_states_values)

        loss = self.model.loss_func(states_action_values, expected_state_action_values)
        loss *= samples.weights
        self.exp_buffer.update_priorities(samples.indices, loss.detach().cpu().numpy())
        self.parameters['experience_beta'].update()
        return loss.mean()

    def endEpisode(self):
        self.log("parameters/beta", self.parameters['experience_beta'].get())
        self.log("time/sample_time", self.sample_time / self.count)
        self.sample_time = 0
        self.count = 1
        super().endEpisode()