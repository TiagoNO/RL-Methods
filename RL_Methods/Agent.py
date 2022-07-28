from distutils.log import Log
import numpy as np
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Logger import Logger
import gym

class Agent:

    def __init__(self, callbacks : Callback = None, logger : Logger = None, log_freq=1) -> None:
        self.callbacks = callbacks
        if self.callbacks is None:
            self.callbacks = Callback()

        self.logger = logger
        if not self.logger is None:
            logger.log("parameters/log_freq", log_freq)

        self.log_freq = log_freq

    def beginTrainning(self):
        pass

    def endTrainning(self):
        pass

    def beginEpisode(self, state):
        pass

    def endEpisode(self):
        pass

    def update(self, state, action, reward, done, next_state, info):
        pass

    def getAction(self, state, deterministic=True, mask=None):
        pass

    def print(self):
        print("\n\n")
        print("|" + "=" * 44 + "|")
        print("|Agent\t|".expandtabs(45))
        print("|Episode {}\t|".format(self.num_episodes+1).expandtabs(45))
        print("|Time steps {}/{}\t|".format(self.num_timesteps, self.total_timesteps).expandtabs(45))
        print("|Episode Score {}\t|".format(self.scores[self.num_episodes]).expandtabs(45))
        print("|Avg score {}\t|".format(round(np.mean(self.scores[-100:]), 2)).expandtabs(45))
        print("|" + "=" * 44 + "|")

    def train(self, env : gym.Env, total_timesteps : int):
        self.total_timesteps = int(total_timesteps)
        self.scores = []
        self.num_timesteps = 0
        self.num_episodes = 0

        self.beginTrainning()
        self.callbacks.beginTrainning()
        
        while self.num_timesteps < total_timesteps:
            obs = env.reset()
            done = False
            action_mask = None
            score = 0
            self.beginEpisode(obs)
            self.callbacks.beginEpisode()

            while not done:
                action = self.getAction(obs, mask=action_mask)
                obs_, reward, done, info = env.step(action)
                
                if 'mask' in info:
                    action_mask = info['mask']

                self.update(obs, action, reward, done, obs_, info)                
                self.callbacks.update()
                
                score += reward
                obs = obs_     
                self.num_timesteps += 1
                
            self.scores.append(score)
            self.print()
            self.endEpisode()
            self.callbacks.endEpisode()
            self.num_episodes+=1

            if not self.logger is None:
                self.logger.log("train/avg_ep_rewards", np.mean(self.scores[-100:]))
                self.logger.dump()

        self.endTrainning()
        self.callbacks.endTrainning()

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

    def save(self, filename):
        pass

    def load(self, filename):
        pass