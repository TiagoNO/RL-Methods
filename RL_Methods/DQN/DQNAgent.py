from json import load
import os
import gym
import pickle
import numpy as np
import torch as th

from RL_Methods.Agent import Agent
from RL_Methods.DQN.DQNModel import DQNModel
from RL_Methods.Buffers.ExperienceBuffer import ExperienceBuffer
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger
from RL_Methods.utils.Common import unzip_files, zip_files

class DQNAgent(Agent):

    def __init__(self,
                    input_dim : gym.Space, 
                    action_dim : gym.Space, 
                    learning_rate : Schedule,
                    epsilon : Schedule,
                    gamma : float, 
                    batch_size : int, 
                    experience_buffer_size : int,
                    target_network_sync_freq : int, 
                    grad_norm_clip:float = 1,
                    architecture=None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    log_freq: int = 1,
                    save_log_every=100,
                    device='cpu',
                    debug=False
                ):        

        super(DQNAgent, self).__init__(
            callbacks=callbacks, 
            logger=logger, 
            log_freq=log_freq, 
            save_log_every=save_log_every,
            debug=debug
            )

        self.parameters['input_dim'] = input_dim
        self.parameters['action_dim'] = action_dim
        self.parameters['learning_rate'] = learning_rate
        self.parameters['epsilon'] = epsilon
        self.parameters['gamma'] = gamma
        self.parameters['batch_size'] = batch_size
        self.parameters['experience_buffer_size'] = experience_buffer_size
        self.parameters['target_network_sync_freq'] = target_network_sync_freq
        self.parameters['grad_norm_clip'] = grad_norm_clip
        self.parameters['architecture']= architecture

        self.losses = []

        self.model = DQNModel(input_dim, action_dim, learning_rate, architecture, device)
        self.exp_buffer = ExperienceBuffer(experience_buffer_size, input_dim, device)      

        if not self.callbacks is None:
            self.callbacks.set_agent(self)

    def getAction(self, state, mask=None, deterministic=False) -> int:
        if mask is None:
            mask = np.ones(self.model.action_dim, dtype=np.bool)

        if np.random.rand() < self.parameters['epsilon'].get() and not deterministic:
            prob = np.array(mask, dtype=np.float) / np.sum(mask)
            return np.random.choice(self.model.action_dim, 1, p=prob).item()
        else:            
            state = th.tensor(state, dtype=th.float).to(self.model.device)
            return self.model.predict(state, deterministic, mask=np.invert(mask))

    def calculate_loss(self) -> th.Tensor:
        samples = self.exp_buffer.sample(self.parameters['batch_size'])

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.parameters['gamma'] * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def step(self) -> None:
        if len(self.exp_buffer) < self.parameters['batch_size']:
            return

        self.model.train(True)
        self.model.optimizer.zero_grad()        
        loss = self.calculate_loss()
        self.logger.log("train/loss", loss.item())

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.parameters['grad_norm_clip'])
        self.model.optimizer.step()


    def update(self, state, action, reward, done, next_state, info) -> None:
        super().update(state, action, reward, done, next_state, info)
        self.exp_buffer.add(state, action, reward, done, next_state)
        self.step()
        self.parameters['epsilon'].update()
        self.model.update_learning_rate()

        if self.parameters['num_timesteps'] % self.parameters['target_network_sync_freq'] == 0:
            self.model.sync()

    def save(self, directory, prefix="dqn", save_exp_buffer=True) -> None:
        if not os.path.isdir(directory):
            os.makedirs(directory)

        model_file = "{}/{}".format(directory, prefix)
        parameters_file = "{}/{}_parameters".format(directory, prefix)
        buffer_file = "{}/{}_buffer".format(directory, prefix)

        self.model.save(model_file)
        pickle.dump(self.parameters, open(parameters_file, "wb"))
        if save_exp_buffer:
            pickle.dump(self.exp_buffer, open(buffer_file, "wb"))

        zip_files(directory, directory)
        for files in os.listdir(directory):
            os.remove(os.path.join(directory, files))
        os.removedirs(directory)

    @classmethod
    def load(cls, filename, logger=None, callback=None, load_buffer=False, device='cpu') -> Agent:
        directory = filename[:filename.rfind(".zip")]
        unzip_files(filename, directory)

        name = filename[filename.rfind("/")+1:]
        prefix = name[:name.find("_")]

        model_file = "{}/{}".format(directory, prefix)
        parameters_file = "{}/{}_parameters".format(directory, prefix)
        buffer_file = "{}/{}_buffer".format(directory, prefix)

        parameters = dict(pickle.load(open(parameters_file, "rb")))
        try:
            num_episodes = parameters.pop('num_episodes')
            num_timesteps = parameters.pop('num_timesteps')
        except:
            num_episodes = 0
            num_timesteps = 0

        agent = cls(
            **parameters,
            logger=logger,
            callbacks=callback,
            device=device
        )
        agent.parameters['num_timesteps'] = num_timesteps
        agent.parameters['num_episodes'] = num_episodes

        agent.model.load(model_file)
        try:
            if load_buffer:
                agent.exp_buffer = pickle.load(open(buffer_file, "rb"))
                agent.exp_buffer.device = device
        except:
            print("Could not load exp buffer...")

        for files in os.listdir(directory):
            os.remove(os.path.join(directory, files))
        os.removedirs(directory)
        return agent

    def train(self, env : gym.Env, total_timesteps : int):
        self.model.train()
        super().train(env, total_timesteps)

    def test(self, env, n_episodes):
        self.model.eval()
        super().test(env, n_episodes)


    def endEpisode(self):
        if not self.logger is None:
            self.logger.log("parameters/learning_rate", self.parameters['learning_rate'].get())
            self.logger.log("parameters/epsilon", self.parameters['epsilon'].get())
        super().endEpisode()
