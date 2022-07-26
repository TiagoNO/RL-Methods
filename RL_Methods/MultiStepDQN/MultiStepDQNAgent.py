from RL_Methods.DQN.DQNAgent import DQNAgent
import torch as th
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
        self.trajectory_steps = trajectory_steps
        self.trajectory = []

    def beginEpisode(self, state):
        for _ in range(len(self.trajectory)):
            state, action, reward, done, next_state = self.getTrajectory()
            self.exp_buffer.add(state, action, reward, done, next_state)
            self.trajectory.pop(0)

    def getTrajectory(self):
        if len(self.trajectory) == 1:
            return self.trajectory[0]

        state = self.trajectory[0][0]
        action = self.trajectory[0][1]
        reward = 0
        done = self.trajectory[0][3]
        next_state = self.trajectory[-1][4]

        for i in reversed(self.trajectory):
            reward = (reward * self.gamma) + i[2]
        
        return state, action, reward, done, next_state

    def calculate_loss(self):
        samples = self.exp_buffer.sample(self.batch_size)

        # calculating q-values for states
        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)

        # using no grad to avoid updating the target network
        with th.no_grad():

            # getting the maximum q-values for next states (using the target network)
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]

            # Calculating the target values (Q(s_next, a) = 0 if state is terminal)
            expected_state_action_values = samples.rewards + ((~samples.dones) * (self.gamma**self.trajectory_steps) * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def update(self, state, action, reward, done, next_state, info):
        if len(self.trajectory) >= self.trajectory_steps:
            t_state, t_action, t_reward, t_done, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_done, t_next_state)
            self.step()
            self.epsilon.update()
            self.model.update_learning_rate()
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, done, next_state])

        if self.num_timesteps % self.checkpoint_freq  == 0:
            self.save(self.savedir + "dqn_{}_steps.pt".format(self.num_timesteps))

        if self.num_timesteps % self.target_network_sync_freq == 0:
            self.model.sync()
