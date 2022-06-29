from abc import abstractclassmethod
from ast import Raise
from cv2 import determinant
import numpy as np

class Agent:

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def beginEpisode(self, state):
        pass

    @abstractclassmethod
    def endEpisode(self):
        pass

    @abstractclassmethod
    def update(self, state, action, reward, done, next_state, info):
        pass

    @abstractclassmethod
    def getAction(self, state, deterministic=True):
        pass

    def train(self, env, n_episodes):
        scores = []

        steps = 0
        for i in range(n_episodes):
            obs = env.reset()
            done = False
            score = 0
            self.beginEpisode(obs)
            while not done:
                action = self.getAction(obs)
                # print(action)
                obs_, reward, done, info = env.step(action)
                self.update(obs, action, reward, done, obs_, info)
                score += reward
                obs = obs_     
                steps += 1
            scores.append(score)
            print("Episode {} - timesteps {} - score {} - avg score {}".format(i+1, steps, np.round(score, 2), np.mean(scores)))
            self.endEpisode()
        return scores

    def test(self, env, n_episodes):
        scores = []
        for i in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self.getAction(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                score += reward
                env.render()   
            scores.append(score)
            print("Episode {} - score {} - avg score {}".format(i+1, np.round(score, 2), np.mean(scores[-200:])))
        return scores

    @abstractclassmethod
    def save(self, filename):
        pass

    @abstractclassmethod
    def load(self, filename):
        pass