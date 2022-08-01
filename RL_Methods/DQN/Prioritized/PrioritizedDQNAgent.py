from tabnanny import check
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th

from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger

class PrioritizedDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate,
                    epsilon,
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq, 
                    experience_prob_alpha, 
                    experience_beta : Schedule, 
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
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta

        if not self.logger is None:
            self.logger.log("parameters/experience_beta_initial", self.beta.initial_value)
            self.logger.log("parameters/experience_beta_final", self.beta.final_value)
            self.logger.log("parameters/experience_beta_delta", self.beta.delta)

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size, self.beta.get())

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        loss = self.model.loss_func(states_action_values, expected_state_action_values)
        loss *= samples.weights
        self.exp_buffer.update_priorities(samples.indices, loss.detach().cpu().numpy())
        self.beta.update()
        self.logger.log("parameters/experience_beta", self.beta.get())
        return loss.mean()


    def print(self):
        super().print()
        print("| Beta: {}\t|".format(self.beta.get()).expandtabs(45))
        print("|" + "=" * 44 + "|")

    def endEpisode(self):
        super().endEpisode()
