import os
import pickle
import numpy as np
import torch as th

from RL_Methods.Agent import Agent
from RL_Methods.DQN.DQNModel import DQNModel
from RL_Methods.Buffers.ExperienceBuffer import ExperienceBuffer
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger, LogLevel
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
                    device : str = 'cpu',
                    callbacks : Callback = None, 
                    logger : Logger = None, 
                    save_log_every:int=100, 
                    verbose : LogLevel = LogLevel.INFO
                ):        
        super().__init__(
                callbacks=callbacks,
                logger=logger, 
                save_log_every=save_log_every, 
                verbose=verbose
                )
        self.data['parameters']['input_dim'] = input_dim
        self.data['parameters']['action_dim'] = action_dim
        self.data['parameters']['learning_rate'] = learning_rate
        self.data['parameters']['epsilon'] = epsilon
        self.data['parameters']['gamma'] = gamma
        self.data['parameters']['batch_size'] = batch_size
        self.data['parameters']['experience_buffer_size'] = experience_buffer_size
        self.data['parameters']['target_network_sync_freq'] = target_network_sync_freq
        self.data['parameters']['grad_norm_clip'] = grad_norm_clip
        self.data['parameters']['architecture']= architecture

        self.model = DQNModel(input_dim, action_dim, learning_rate, architecture, device)
        self.exp_buffer = ExperienceBuffer(experience_buffer_size, input_dim, device)      

        self.loss_mean = 0
        self.loss_count = 0

    def getAction(self, state, mask=None, deterministic=False) -> int:
        action = -1
        if mask is None:
            mask = np.ones(self.model.action_dim, dtype=bool)

        if np.random.rand() < self.data['parameters']['epsilon'].get() and not deterministic:
            prob = np.array(mask, dtype=np.float32) / np.sum(mask)
            action = np.random.choice(self.model.action_dim, 1, p=prob).item()
        else:
            self.model.train(False)
            state = th.from_numpy(state).to(self.model.device)
            action = self.model.predict(state, deterministic, mask=np.invert(mask))
        return action

    def calculate_loss(self) -> th.Tensor:
        samples = self.exp_buffer.sample(self.data['parameters']['batch_size'])
        dones = th.bitwise_or(samples.terminated, samples.truncated)

        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)
        with th.no_grad():
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]
            expected_state_action_values = samples.rewards + ((~dones) * self.data['parameters']['gamma'] * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def step(self) -> None:
        if len(self.exp_buffer) < self.data['parameters']['batch_size']:
            return

        self.model.train(True)
        self.model.optimizer.zero_grad()
        loss = self.calculate_loss()
        self.loss_mean += loss.item()
        self.loss_count += 1

        loss.backward()
        self.model.clip_grad(self.data['parameters']['grad_norm_clip'])
        self.model.optimizer.step()

    def learn(self):
        self.step()
        self.data['parameters']['epsilon'].update()
        self.model.update_learning_rate()

        if self.data['num_timesteps'] % self.data['parameters']['target_network_sync_freq'] == 0:
            self.model.sync()

    def update(self, state, action, reward, terminated, truncated, next_state, info) -> None:
        self.exp_buffer.add(state, action, reward, terminated, truncated, next_state)

    def save(self, filename, save_exp_buffer=True) -> None:
        directory = os.path.splitext(filename)[0]
        prefix = os.path.basename(directory)

        os.makedirs(directory)

        model_file = "{}/{}_policy.pt".format(directory, prefix)
        parameters_file = "{}/{}_data".format(directory, prefix)
        buffer_file = "{}/{}_buffer".format(directory, prefix)
        callback_file = "{}/{}_callback".format(directory, prefix)
    
        self.model.save(model_file)

        with open(parameters_file, "wb") as params_f_ptr:
            pickle.dump(self.data, params_f_ptr)

        if save_exp_buffer:
            with open(buffer_file, "wb") as buffer_f_ptr:
                pickle.dump(self.exp_buffer, buffer_f_ptr)

        with open(callback_file, "wb") as callback_f_ptr:
            pickle.dump(self.callbacks, callback_f_ptr)

        zip_files(directory, directory)
        for files in os.listdir(directory):
            os.remove(os.path.join(directory, files))
        os.removedirs(directory)

    @classmethod
    def load(cls, filename, logger=None, callback=None, load_buffer=False, device='cpu') -> Agent:
        directory, extension = os.path.splitext(filename)
        prefix = os.path.basename(directory)
        assert extension == '.zip'

        unzip_files(filename, directory)

        model_file = "{}/{}_policy.pt".format(directory, prefix)
        data_file = "{}/{}_data".format(directory, prefix)
        buffer_file = "{}/{}_buffer".format(directory, prefix)
        callback_file = "{}/{}_callback".format(directory, prefix)

        with open(data_file, "rb") as data_f_ptr:
            data = dict(pickle.load(data_f_ptr))

        if callback is None:
            with open(callback_file, "rb") as callback_f_ptr:
                callback = pickle.load(callback_f_ptr)

        agent = cls(
            **data['parameters'],
            logger=logger,
            callbacks=callback,
            device=device
        )
        agent.data['num_timesteps'] = data['num_timesteps']
        agent.data['num_episodes'] = data['num_episodes']

        agent.model.load(model_file)
        if load_buffer:
            with open(buffer_file, "rb") as buffer_f_ptr:
                agent.exp_buffer = pickle.load(open(buffer_file, "rb"))
                agent.exp_buffer.device = device

        for files in os.listdir(directory):
            os.remove(os.path.join(directory, files))
        os.removedirs(directory)
        return agent

    def train(self, env, total_timesteps : int):
        self.model.train()
        super().train(env, total_timesteps)

    def test(self, env, n_episodes):
        self.model.eval()
        super().test(env, n_episodes)


    def endEpisode(self):
        self.log(LogLevel.INFO, "parameters/learning_rate", self.data['parameters']['learning_rate'].get())
        self.log(LogLevel.INFO, "parameters/epsilon", self.data['parameters']['epsilon'].get())
        if(self.loss_count > 0):
            self.log(LogLevel.INFO, "train/loss_mean", self.loss_mean / self.loss_count)

        self.loss_mean = 0
        self.loss_count = 0

        super().endEpisode()
