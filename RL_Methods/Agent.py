from abc import abstractclassmethod
from ast import Raise
from cv2 import determinant
import numpy as np

class Agent:

    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def beginTrainning(self):
        pass

    @abstractclassmethod
    def endTrainning(self):
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

    def print(self):
        print("|" + "=" * 44 + "|")
        print("|Agent\t|".format(self.num_episodes+1).expandtabs(45))
        print("|Episode {}/{}\t|".format(self.num_episodes+1, self.total_episodes).expandtabs(45))
        print("|Time steps {}\t|".format(self.num_timesteps).expandtabs(45))
        print("|Episode Score {}\t|".format(self.scores[self.num_episodes]).expandtabs(45))
        print("|Avg score {}\t|".format(round(np.mean(self.scores[max(0, self.num_episodes-100):self.num_episodes+1]), 2)).expandtabs(45))
        print("|" + "=" * 44 + "|")

    def train(self, env, n_episodes):
        self.scores = np.zeros(n_episodes, dtype=np.float32)
        self.num_timesteps = 0
        self.num_episodes = 0
        self.total_episodes = n_episodes

        self.beginTrainning()
        for self.num_episodes in range(self.total_episodes):
            obs = env.reset()
            done = False
            score = 0
            self.beginEpisode(obs)
            while not done:
                action = self.getAction(obs)
                obs_, reward, done, info = env.step(action)
                self.update(obs, action, reward, done, obs_, info)
                score += reward
                obs = obs_     
                self.num_timesteps += 1
            self.scores[self.num_episodes] = score
            print("\n\n")
            self.print()
            self.endEpisode()
        self.endTrainning()

    def test(self, env, n_episodes):
        self.total_test_episodes = n_episodes
        self.num_test_episodes = 0

        self.test_scores = np.zeros(n_episodes, dtype=np.float32)

        for self.num_test_episodes in range(self.total_test_episodes):
            obs = env.reset()
            done = False
            score = 0
            while not done:
                action = self.getAction(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                score += reward
                env.render()   
            self.test_scores[self.num_test_episodes] = score
            print("|" + "=" * 44 + "|")
            print("|Agent\t|".format(self.num_episodes+1).expandtabs(45))
            print("|Episode {}/{}\t|".format(self.num_test_episodes+1, self.total_test_episodes).expandtabs(45))
            print("|Episode Score {}\t|".format(score).expandtabs(45))
            print("|Avg score {}\t|".format(round(np.mean(self.test_scores[0:self.num_test_episodes+1]), 2)).expandtabs(45))
            print("|" + "=" * 44 + "|")
    @abstractclassmethod
    def save(self, filename):
        pass

    @abstractclassmethod
    def load(self, filename):
        pass