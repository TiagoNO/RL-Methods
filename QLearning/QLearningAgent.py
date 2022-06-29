from functools import update_wrapper
from Agent import Agent
import numpy as np

class QLearningAgent (Agent):

    def __init__(self, n_actions, lr=0.1, discount=0.9, initial_epsilon=1.0, final_epsilon=.1, epsilon_decay=1e-3):
        self.q_table = {}
        self.lr = lr
        self.discount = discount
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.k = 0

    def initializeQValue(self, state):
        if not state in self.q_table:
            self.q_table[state] = {}
            for a in range(self.n_actions):
                self.q_table[state][a] = 0

    def getQValue(self, state, action):
        return self.q_table[state][action]

    def getBestQValue(self, state):
        best = -999999
        for a in range(self.n_actions):
            best = max(best, self.getQValue(state, a))
        return best

    def getBestAction(self, state):
        best = -999999
        action = 0
        for a in range(self.n_actions):
            value = self.getQValue(state, a)
            if best < value:
                best = value
                action = a
        return action

    def getAction(self, state):
        state_n = state.__str__()
        self.initializeQValue(state_n)

        random = np.random.rand()
        if random < self.epsilon:
            self.last_action = np.random.choice(range(self.n_actions))
        else:
            self.last_action = self.getBestAction(state_n)
        self.k += 1
        return self.last_action

    def update(self, state, action, reward, done, next_state, info):
        state_n = state.__str__()
        new_state_n = next_state.__str__()

        self.initializeQValue(state_n)
        self.initializeQValue(new_state_n)

        next_q_value = self.getBestQValue(new_state_n)
        self.q_table[state_n][action] = (1 - self.lr) * self.q_table[state_n][action] + self.lr * (reward + self.discount * next_q_value)
        self.update_epsilon()

    def update_epsilon(self):
        self.epsilon = max(self.final_epsilon, 1 - self.k * self.epsilon_decay)

class DoubleQLearningAgent(QLearningAgent):

    def __init__(self, n_actions, lr=0.1, discount=0.9, initial_epsilon=1.0, final_epsilon=.1, epsilon_decay=1e-3):
        super().__init__(n_actions, lr, discount, initial_epsilon, final_epsilon, epsilon_decay)
        self.q_table_b = {}

    def initializeQValue(self, state):
        if not state in self.q_table:
            self.q_table[state] = {}
            for a in range(self.n_actions):
                self.q_table[state][a] = (2 * np.random.rand()) - 1

        if not state in self.q_table_b:
            self.q_table_b[state] = {}
            for a in range(self.n_actions):
                self.q_table_b[state][a] = (2 * np.random.rand()) - 1

    def getQValue(self, table, state, action):
        return table[state][action]

    def getBestQValue(self, table, state):
        best = -999999
        for a in range(self.n_actions):
            best = max(best, self.getQValue(table, state, a))
        return best

    def getBestAction(self, table, state):
        best = -999999
        action = 0
        for a in range(self.n_actions):
            value = self.getQValue(table, state, a)
            if best < value:
                best = value
                action = a
        return action

    def getBestActionAB(self, state):
        best = 0
        action = 0
        for a in range(self.n_actions):
            value_a = self.getQValue(self.q_table, state, a)
            value_b = self.getQValue(self.q_table_b, state, a)
            if best < value_a:
                best = value_a
                action = a

            if best < value_b:
                best = value_b
                action = a

        return action

    def getAction(self, state):
        state_n = state.__str__()
        self.initializeQValue(state_n)

        random = np.random.rand()
        if random < self.epsilon:
            action = np.random.choice(range(self.n_actions))
        else:
            action = self.getBestActionAB(state_n)
        self.k += 1
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
        self.update_epsilon()