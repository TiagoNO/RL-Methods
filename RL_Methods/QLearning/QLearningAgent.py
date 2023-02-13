from RL_Methods.Agent import Agent
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger, LogLevel

import numpy as np

class QLearningAgent (Agent):

    def __init__(self, 
                n_actions : int,
                learning_rate : Schedule,
                discount: Schedule,
                epsilon : Schedule,
                callbacks : Callback = None, 
                logger : Logger = None, 
                save_log_every : int = 100, 
                verbose: LogLevel = LogLevel.INFO
                ):
        super().__init__(callbacks, logger, save_log_every, verbose)
        self.q_table = {}
        self.lr = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.n_actions = n_actions

    def initializeQValue(self, state):
        if not state in self.q_table:
            self.q_table[state] = (2 * np.random.rand(self.n_actions)) - 1

    def getQValue(self, state, action):
        return self.q_table[state][action]

    def getBestQValue(self, state):
        return np.max(self.q_table[state])

    def getBestAction(self, state):
        return np.argmax(self.q_table[state])

    def getAction(self, state, mask=None, deterministic=False):
        state_n = state.__str__()
        self.initializeQValue(state_n)

        random = np.random.rand()
        if random < self.epsilon.get():
            self.last_action = np.random.choice(range(self.n_actions))
        else:
            self.last_action = self.getBestAction(state_n)
        return self.last_action

    def update(self, state, action, reward, terminated, truncated, next_state, info):
        state_n = state.__str__()
        new_state_n = next_state.__str__()

        self.initializeQValue(state_n)
        self.initializeQValue(new_state_n)

        next_q_value = self.getBestQValue(new_state_n)
        self.q_table[state_n][action] = (1 - self.lr.get()) * self.q_table[state_n][action] + self.lr.get() * (reward + self.discount * next_q_value)
        self.logger.log(LogLevel.INFO, "epsilon", self.epsilon.get())
        self.logger.log(LogLevel.INFO, "learning_rate", self.lr.get())
        self.epsilon.update()
        self.lr.update()