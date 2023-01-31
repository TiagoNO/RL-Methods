import os
import gym
import pickle
import numpy as np
import torch as th
import time

from RL_Methods.Agent import Agent
from RL_Methods.DQN.DQNModel import DQNModel
from RL_Methods.Buffers.ExperienceBuffer import ExperienceBuffer
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger
from RL_Methods.utils.Common import unzip_files, zip_files

class DQNAgent(Agent):

    def __init__(self,
                    input_dim : tuple, 
                    action_dim : int, 
                    learning_rate : Schedule,
                    epsilon : Schedule,
                    gamma : float, 
                    batch_size : int, 
                    experience_buffer_size : int,
                    target_network_sync_freq : int, 
                    grad_norm_clip: float = 1,
                    architecture: dict = None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    save_log_every: int = 100,
                    device: str = 'cpu',
                    verbose: int = 0
                ):        

        super(DQNAgent, self).__init__(
            callbacks=callbacks, 
            logger=logger, 
            save_log_every=save_log_every,
            verbose=verbose
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

        self.model = DQNModel(input_dim, action_dim, learning_rate, architecture, device)
        self.exp_buffer = ExperienceBuffer(experience_buffer_size, input_dim, device)      

        self.loss_mean = 0
        self.loss_count = 0

        if not self.callbacks is None:
            self.callbacks.set_agent(self)

    def getAction(self, state, mask=None, deterministic=False) -> int:
        if mask is None:
            mask = np.ones(self.model.action_dim, dtype=bool)

        if np.random.rand() < self.parameters['epsilon'].get() and not deterministic:
            prob = np.array(mask, dtype=np.float32) / np.sum(mask)
            return np.random.choice(self.model.action_dim, 1, p=prob).item()
        else:
            state = th.from_numpy(state).to(self.model.device)
            action = self.model.predict(state, deterministic, mask=np.invert(mask))
            return action

    def calculate_loss(self) -> th.Tensor:
        samples = self.exp_buffer.sample(self.parameters['batch_size'])
        dones = th.bitwise_or(samples.terminated, samples.truncated)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~dones) * self.parameters['gamma'] * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def step(self) -> None:
        if len(self.exp_buffer) < self.parameters['batch_size']:
            return

        self.model.train(True)
        self.model.optimizer.zero_grad()
        loss = self.calculate_loss()
        self.loss_mean += loss.item()
        self.loss_count += 1

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.parameters['grad_norm_clip'])
        self.model.optimizer.step()

    def learn(self):
        self.step()
        self.parameters['epsilon'].update()
        self.model.update_learning_rate()

        if self.parameters['num_timesteps'] % self.parameters['target_network_sync_freq'] == 0:
            self.model.sync()

    def update(self, state, action, reward, terminated, truncated, next_state, info) -> None:
        self.exp_buffer.add(state, action, reward, terminated, truncated, next_state)

    def save(self, directory, prefix="dqn", save_exp_buffer=True) -> None:
        if not os.path.isdir(directory):
            os.makedirs(directory)

        model_file = "{}/{}".format(directory, prefix)
        parameters_file = "{}/{}_parameters".format(directory, prefix)
        buffer_file = "{}/{}_buffer".format(directory, prefix)
        callback_file = "{}/{}_callback".format(directory, prefix)
    
        self.model.save(model_file)
        params_f_ptr = open(parameters_file, "wb")
        pickle.dump(self.parameters, params_f_ptr)
        params_f_ptr.close()

        if save_exp_buffer:
            buffer_f_ptr = open(buffer_file, "wb")
            pickle.dump(self.exp_buffer, buffer_f_ptr)
            buffer_f_ptr.close()

        callback_f_ptr = open(callback_file, "wb")
        pickle.dump(self.callbacks, callback_f_ptr)
        callback_f_ptr.close()

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
        callback_file = "{}/{}_callback".format(directory, prefix)

        parameters_f_ptr = open(parameters_file, "rb")
        parameters = dict(pickle.load(parameters_f_ptr))
        parameters_f_ptr.close()

        try:
            if callback is None:
                callback_f_ptr = open(callback_file, "rb")
                callback = pickle.load(callback_f_ptr)
                callback_f_ptr.close()
        except Exception as e:
            print("Could not load callbacks: {}".format(e))

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
                buffer_f_ptr = open(buffer_file, "rb")
                agent.exp_buffer = pickle.load(open(buffer_file, "rb"))
                agent.exp_buffer.device = device
                buffer_f_ptr.close()
        except Exception as e:
            print("Could not load exp buffer: {}".format(e))

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
        self.log("parameters/learning_rate", self.parameters['learning_rate'].get())
        self.log("parameters/epsilon", self.parameters['epsilon'].get())
        if(self.loss_count > 0):
            self.log("train/loss_mean", self.loss_mean / self.loss_count)

        self.loss_mean = 0
        self.loss_count = 0

        super().endEpisode()
