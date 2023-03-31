import os
import numpy as np
from RL_Methods.utils.Logger import LogLevel

class Callback:

    def __init__(self) -> None:
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    def beginTrainning(self):
        pass

    def endTrainning(self):
        pass

    def beginEpisode(self):
        pass

    def endEpisode(self):
        pass

    def update(self):
        pass

class ListCallback(Callback):
    
    def __init__(self, callback_list : list) -> None:
        super().__init__()
        self.list = callback_list
    
    def add(self, callback : Callback):
        self.list.append(callback)

    def set_agent(self, agent):
        for callback in self.list:
            callback.set_agent(agent)

    def beginTrainning(self):
        for callback in self.list:
            callback.beginTrainning()

    def endTrainning(self):
        for callback in self.list:
            callback.endTrainning()

    def beginEpisode(self):
        for callback in self.list:
            callback.beginEpisode()

    def endEpisode(self):
        for callback in self.list:
            callback.endEpisode()

    def update(self):
        for callback in self.list:
            callback.update()

class AgentStatisticsCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.scores = []
        self.ep_scores = []

    def update(self):
        self.scores.append(self.agent.data['episode']['reward'])

    def endEpisode(self):
        score =  np.sum(self.scores)
        if len(self.ep_scores) >= 50:
            self.ep_scores.pop(0)
        self.ep_scores.append(score)

        self.agent.log(LogLevel.INFO, "train/avg_ep_rewards", np.mean(self.ep_scores))
        self.agent.log(LogLevel.INFO, "train/ep_score", score)
        self.scores.clear()



class CheckpointCallback (Callback):

    def __init__(self, filename, directory, checkpoint_freq, save_buffer=False) -> None:        
        self.directory = directory
        self.filename = os.path.join(directory, os.path.splitext(filename)[0])
        self.checkpoint_freq = checkpoint_freq
        self.save_buffer = save_buffer

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    def update(self):   
        if self.agent.data['num_timesteps'] % self.checkpoint_freq == 0 and self.agent.data['num_timesteps'] > 0:
            self.agent.save("{}_{}_steps".format(self.filename, self.agent.data['num_timesteps']), save_exp_buffer=self.save_buffer)

    def endTrainning(self):
        self.agent.save("{}_{}_steps".format(self.filename, self.agent.data['num_timesteps']), save_exp_buffer=self.save_buffer)