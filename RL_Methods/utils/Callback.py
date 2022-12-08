import os
import numpy as np

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
        self.scores.append(self.agent.data['reward'])

    def endEpisode(self):
        score =  np.sum(self.scores)
        self.ep_scores.append(score)
        self.agent.logger.log("train/avg_ep_rewards", np.mean(self.ep_scores[-50:]))
        self.agent.logger.update("train/ep_score", score)
        self.scores.clear()



class CheckpointCallback (Callback):

    def __init__(self, savedir, prefix, checkpoint_freq, save_buffer=False) -> None:
        self.savedir = savedir
        self.prefix = prefix
        self.checkpoint_freq = checkpoint_freq
        self.save_buffer = save_buffer

        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

    def beginTrainning(self):
        f = open(self.savedir + self.prefix + "_experiment_info.txt", "w")
        f.write(self.agent.__str__())
        f.close()

    def update(self):
        if self.agent.parameters['num_timesteps'] % self.checkpoint_freq == 0 and self.agent.parameters['num_timesteps'] > 0:
            self.agent.save(self.savedir + self.prefix + "_{}_steps".format(self.agent.parameters['num_timesteps']), prefix=self.prefix, save_exp_buffer=self.save_buffer)

    def endTrainning(self):
        self.agent.save(self.savedir + self.prefix + "_last", prefix=self.prefix, save_exp_buffer=True)