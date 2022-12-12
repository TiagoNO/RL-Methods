from distutils.log import Log
import numpy as np
from RL_Methods.utils.Callback import Callback, AgentStatisticsCallback, ListCallback
from RL_Methods.utils.Logger import Logger
import gym
import time

from RL_Methods.utils.Schedule import Schedule

class Agent:

    def __init__(self, callbacks : Callback = None, logger : Logger = None, log_freq:int=1, save_log_every:int=100, verbose:int=0) -> None:
        if logger is None:
            self.logger = Logger(None)
        else:
            self.logger = logger

        if callbacks is None:
            self.callbacks = AgentStatisticsCallback()
        else:
            self.callbacks = ListCallback([AgentStatisticsCallback(), callbacks])

        self.parameters = {}
        self.parameters['log_freq'] = log_freq
        self.parameters['save_log_every'] = save_log_every
        self.parameters['verbose'] = verbose
        self.parameters['num_timesteps'] = 0
        self.parameters['num_episodes'] = 1
        self.data = {
            # curr data
            'state':None,
            'action':None,
            'reward':None,
            'done':False,
            'next_state':None,
            'info':None,
        }

    def __str__(self) -> str:
        params = "{}:\n".format(self.__class__.__name__)
        for p in self.parameters:
            params += "{}: {}\n".format(p, self.parameters[p])
        return params

    def beginTrainning(self):
        pass

    def endTrainning(self):
        pass

    def beginEpisode(self, state):
        pass

    def endEpisode(self):
        if not self.logger is None:
            self.logger.log("train/timesteps", self.parameters['num_timesteps'])
            self.logger.log("train/episodes", self.parameters['num_episodes'])
            self.logger.print()
            if self.parameters['num_episodes'] % self.parameters['save_log_every'] == 0:
                self.logger.dump()

    def update(self, state, action, reward, done, next_state, info):
        pass

    def getAction(self, state, deterministic=True, mask=None):
        pass

    def learn(self):
        pass

    def train(self, env : gym.Env, total_timesteps : int, reset=False):
        self.total_timesteps = int(total_timesteps)

        self.beginTrainning()
        self.callbacks.beginTrainning()
        
        while self.parameters['num_timesteps'] < total_timesteps:
            obs = env.reset()
            done = False
            action_mask = None
            score = 0
            self.beginEpisode(obs)
            self.callbacks.beginEpisode()
            while not done:
                action = self.getAction(obs, mask=action_mask)
                obs_, reward, done, info = env.step(action)
                self.learn()
                
                if 'mask' in info:
                    action_mask = info['mask']

                self.update(obs, action, reward, done, obs_, info)
                self.data['state'] = obs
                self.data['action'] = action
                self.data['reward'] = reward
                self.data['done'] = done
                self.data['next_state'] = obs_
                self.data['info'] = info

                self.callbacks.update()


                score += reward
                obs = obs_     
                self.parameters['num_timesteps'] += 1

            self.callbacks.endEpisode()
            self.parameters['num_episodes'] += 1

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