from RL_Methods.QLearning.QLearningAgent import QLearningAgent
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger, LogLevel
from RL_Methods.utils.Schedule import Schedule
import numpy as np

class DoubleQLearningAgent(QLearningAgent):

    def __init__(self, 
                n_actions: int, 
                learning_rate: Schedule, 
                discount: Schedule, 
                epsilon: Schedule, 
                callbacks: Callback = None, 
                logger: Logger = None, 
                save_log_every: int = 100, 
                verbose: LogLevel = LogLevel.INFO
                ):
        super().__init__(n_actions, learning_rate, discount, epsilon, callbacks, logger, save_log_every, verbose)
        self.q_table_b = {}

    def initializeQValue(self, state):
        if not state in self.q_table:
            self.q_table[state] = (2 * np.random.rand(self.n_actions)) - 1

        if not state in self.q_table_b:
            self.q_table_b[state] = (2 * np.random.rand()) - 1

    def getQValue(self, table, state, action):
        return table[state][action]

    def getBestQValue(self, table, state):
        return np.max(table[state])

    def getBestAction(self, table, state):
        return np.argmax(table[state])

    def getBestActionAB(self, state):
        value_a = self.getBestQValue(self.q_table, state)
        value_b = self.getBestQValue(self.q_table_b, state)
        if(value_a > value_b):
            return self.getBestAction(self.q_table, state)
        else:
            return self.getBestAction(self.q_table_b, state)

    def getAction(self, state, mask=None, deterministic=False):
        state_n = state.__str__()
        self.initializeQValue(state_n)

        random = np.random.rand()
        if random < self.epsilon.get():
            action = np.random.choice(range(self.n_actions))
        else:
            action = self.getBestActionAB(state_n)
        return action

    def update(self, state, action, reward, new_state, done):
        state_n = state.__str__()
        new_state_n = new_state.__str__()

        update_a_prob = np.random.rand()
        if update_a_prob < .5:
            self.initializeQValue(state_n)
            self.initializeQValue(new_state_n)

            best_action = self.getBestAction(self.q_table, new_state_n)
            next_q_value = self.getQValue(self.q_table_b, new_state_n, best_action)
            self.q_table[state_n][action] += self.lr * (reward + self.discount * next_q_value - self.getQValue(self.q_table, state_n, action))
        else:
            self.initializeQValue(state_n)
            self.initializeQValue(new_state_n)

            best_action = self.getBestAction(self.q_table_b, new_state_n)
            next_q_value = self.getQValue(self.q_table, new_state_n, best_action)
            self.q_table_b[state_n][action] += self.lr * (reward + self.discount * next_q_value - self.getQValue(self.q_table_b, state_n, action))
        self.logger.log(LogLevel.INFO, "epsilon", self.epsilon.get())
        self.logger.log(LogLevel.INFO, "learning_rate", self.lr.get())
        self.epsilon.update()
        self.lr.update()