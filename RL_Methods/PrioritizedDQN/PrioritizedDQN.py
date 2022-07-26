from tabnanny import check
from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.Buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import torch as th

from RL_Methods.utils.Schedule import Schedule

class PrioritizedDQN(DQNAgent):

    def __init__(self, input_dim, 
                       action_dim, 
                       learning_rate,
                       epsilon,
                       gamma, 
                       batch_size, 
                       experience_buffer_size, 
                       target_network_sync_freq, 
                       experience_prob_alpha, 
                       experience_beta : Schedule, 
                       checkpoint_freq,
                       savedir,
                       log_freq,
                       architecture,
                       device='cpu'):
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
        self.exp_buffer = PrioritizedReplayBuffer(experience_buffer_size, input_dim, device, experience_prob_alpha)
        self.beta = experience_beta

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
        return loss.mean()


    def print(self):
        super().print()
        print("| Beta: {}\t|".format(self.beta.get()).expandtabs(45))
        print("|" + "=" * 44 + "|")

    def endEpisode(self):
        super().endEpisode()
