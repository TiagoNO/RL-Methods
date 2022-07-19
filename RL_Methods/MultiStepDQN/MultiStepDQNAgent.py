from RL_Methods.DQN.DQNAgent import DQNAgent

class MultiStepDQNAgent(DQNAgent):

    def __init__(self, 
                    input_dim, 
                    action_dim, 
                    initial_epsilon, 
                    final_epsilon, 
                    epsilon_decay, 
                    learning_rate, 
                    gamma, 
                    batch_size, 
                    experience_buffer_size, 
                    target_network_sync_freq,
                    trajectory_steps,
                    device='cpu'
                ):
        
        super().__init__(input_dim, action_dim, initial_epsilon, final_epsilon, epsilon_decay, learning_rate, gamma, batch_size, experience_buffer_size, target_network_sync_freq, device)
        self.trajectory_steps = trajectory_steps
        self.trajectory = []

    def beginEpisode(self, state):
        for i in range(len(self.trajectory)):
            state, action, reward, done, next_state = self.getTrajectory()
            self.exp_buffer.add(state, action, reward, done, next_state)
            self.trajectory.pop(0)

    def getTrajectory(self):
        state = self.trajectory[0][0]
        action = self.trajectory[0][1]
        reward = 0
        done = self.trajectory[0][3]
        next_state = self.trajectory[-1][4]

        for i in reversed(self.trajectory):
            reward = (reward * self.gamma) + i[2]
        
        return state, action, reward, done, next_state

    def update(self, state, action, reward, done, next_state, info):
        if len(self.trajectory) >= self.trajectory_steps:
            t_state, t_action, t_reward, t_done, t_next_state = self.getTrajectory()
            self.exp_buffer.add(t_state, t_action, t_reward, t_done, t_next_state)
            self.updateEpsilon()
            self.step()
            self.trajectory.pop(0)

        self.trajectory.append([state, action, reward, done, next_state])

        if self.num_timesteps % self.target_network_sync_freq == 0:
            self.model.sync()
