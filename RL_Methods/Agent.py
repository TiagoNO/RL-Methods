from distutils.log import Log
import numpy as np
from RL_Methods.utils.Callback import Callback, AgentStatisticsCallback, ListCallback
from RL_Methods.utils.Logger import Logger, LogLevel
import gymnasium as gym
import time

from RL_Methods.utils.Schedule import Schedule

class Agent:

    def __init__(self, 
                callbacks : Callback = None, 
                logger : Logger = None, 
                save_log_every:int=100, 
                verbose : LogLevel = LogLevel.INFO
                ):

        if logger is None:
            self.logger = Logger(None)
        else:
            self.logger = logger
        self.logger.setLevel(verbose)

        if callbacks is None:
            self.callbacks = AgentStatisticsCallback()
        else:
            self.callbacks = ListCallback([AgentStatisticsCallback(), callbacks])
        self.callbacks.set_agent(self)

        self.data = {
            'parameters':{
                'save_log_every' : save_log_every,
                'verbose' : verbose,
            },

            'episode':{
                'state':None,
                'start_info':None,
                'action':None,
                'reward':None,
                'terminated':False,
                'truncated':False,
                'next_state':None,
                'info':None
            },

            'num_timesteps' : 0,
            'num_episodes' : 0
        }

    def log(self, level, name, value):
        self.logger.log(level, name, value)

    def __str__(self) -> str:
        params = "{}:\n".format(self.__class__.__name__)
        for p in self.data['parameters']:
            params += "{}: {}\n".format(p, self.data['parameters'][p])
        return params

    def beginTrainning(self):
        pass

    def endTrainning(self):
        pass

    def beginEpisode(self):
        pass

    def endEpisode(self):
        self.log(LogLevel.INFO, "train/episodes", self.data['num_episodes'])
        self.log(LogLevel.INFO, "train/timesteps", self.data['num_timesteps'])
        self.logger.print()
        if self.data['num_episodes'] % self.data['parameters']['save_log_every'] == 0:
            self.logger.dump()

    def update(self, state, action, reward, terminated, truncated, next_state, info):
        pass

    def getAction(self, state, deterministic=True, mask=None):
        pass

    def learn(self):
        pass

    def train(self, env : gym.Env, total_timesteps : int, reset=False):
        self.total_timesteps = int(total_timesteps)

        self.beginTrainning()
        self.callbacks.beginTrainning()
        
        while self.data['num_timesteps'] < total_timesteps:
            obs, start_info = env.reset()

            done = False
            action_mask = None
            score = 0

            self.data['episode']['state'] = obs
            self.data['episode']['start_info'] = start_info

            self.beginEpisode()
            self.callbacks.beginEpisode()
            while not done:
                action = self.getAction(obs, mask=action_mask)
                obs_, reward, terminated, truncated, info = env.step(action)
                done = (terminated or truncated)
                self.learn()
                
                if 'mask' in info:
                    action_mask = info['mask']

                self.update(obs, action, reward, terminated, truncated, obs_, info)
                self.data['episode']['state'] = obs
                self.data['episode']['action'] = action
                self.data['episode']['reward'] = reward
                self.data['episode']['terminated'] = terminated
                self.data['episode']['truncated'] = truncated
                self.data['episode']['next_state'] = obs_
                self.data['episode']['info'] = info

                self.callbacks.update()


                score += reward
                obs = obs_     
                self.data['num_timesteps'] += 1

            self.callbacks.endEpisode()
            self.data['num_episodes'] += 1

            self.endEpisode()

        self.callbacks.endTrainning()
        self.endTrainning()
        self.logger.dump()

    def test(self, env, n_episodes):
        self.total_test_episodes = n_episodes
        self.num_test_episodes = 0

        self.test_scores = np.zeros(n_episodes, dtype=np.float32)

        for self.num_test_episodes in range(self.total_test_episodes):
            obs = env.reset()
            done = False
            action_mask = None
            score = 0
            while not done:
                action = self.getAction(obs, deterministic=True, mask=action_mask)
                obs, reward, done, info = env.step(action)
                if 'mask' in info:
                    action_mask = info['mask']

                score += reward
                env.render()   
            self.test_scores[self.num_test_episodes] = score

    def save(self, filename):
        pass

    @staticmethod
    def load(self, filename):
        pass