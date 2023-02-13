import torch as th

from RL_Methods.DQN.DQNAgent import DQNAgent
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger, LogLevel


class MultiStepDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    learning_rate,
                    epsilon,
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    trajectory_steps,
                    architecture,
                    grad_norm_clip=1,
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
                        architecture=architecture,
                        callbacks=callbacks,
                        logger=logger,
                        save_log_every=save_log_every,
                        device=device,
                        verbose=verbose
                        )

        self.trajectory = []
        self.parameters['trajectory_steps'] = trajectory_steps

    def beginEpisode(self):
        while(len(self.trajectory) > 0):
            state, action, reward, terminated, truncated, next_state = self.getTrajectory()
            self.exp_buffer.add(state, action, reward, terminated, truncated, next_state)
            self.trajectory.pop(0)

    def getTrajectory(self):
        if len(self.trajectory) == 1:
            return self.trajectory[0]

        state = self.trajectory[0][0]
        action = self.trajectory[0][1]
        reward = 0
        terminated = self.trajectory[0][3]
        truncated = self.trajectory[0][4]
        next_state = self.trajectory[-1][5]

        for i in reversed(self.trajectory):
            reward = (reward * self.parameters['gamma']) + i[2]
        
        return state, action, reward, terminated, truncated, next_state

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.parameters['batch_size'])
        dones = th.bitwise_or(samples.terminated, samples.truncated)
        
        # calculating q-values for states
        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)

        # using no grad to avoid updating the target network
        with th.no_grad():

            # getting the maximum q-values for next states (using the target network)
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]

            # Calculating the target values (Q(s_next, a) = 0 if state is terminal)
            gamma = self.parameters['gamma']**self.parameters['trajectory_steps']
            expected_state_action_values = samples.rewards + ((~dones) * gamma * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def update(self, state, action, reward, terminated, truncated, next_state, info):
        if len(self.trajectory) >= self.parameters['trajectory_steps']:
            t_state, t_action, t_reward, t_terminated, t_truncated, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_terminated, t_truncated, t_next_state)
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, terminated, truncated, next_state])