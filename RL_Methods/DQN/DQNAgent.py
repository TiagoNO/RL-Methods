import os
import gym
import pickle
import numpy as np
import torch as th

from RL_Methods.Agent import Agent
from RL_Methods.DQN.Model import Model
from RL_Methods.Buffers.ExperienceBuffer import ExperienceBuffer
from RL_Methods.utils.Callback import Callback
from RL_Methods.utils.Schedule import Schedule
from RL_Methods.utils.Logger import Logger

class DQNAgent(Agent):

    def __init__(self, 
                    input_dim : gym.Space, 
                    action_dim : gym.Space, 
                    learning_rate: Schedule,
                    epsilon : Schedule,
                    gamma : float, 
                    batch_size : int, 
                    experience_buffer_size : int,
                    target_network_sync_freq : int, 
                    grad_norm_clip=1,
                    architecture=None,
                    callbacks: Callback = None,
                    logger: Logger = None,
                    log_freq: int = 1,
                    save_log_every=100,
                    device='cpu'
                ):        
        super().__init__(callbacks, logger, log_freq, save_log_every)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.architecture = architecture

        self.num_timesteps = 0
        self.target_network_sync_freq = target_network_sync_freq

        self.model = self.create_model(learning_rate, architecture, device)
        self.exp_buffer = ExperienceBuffer(experience_buffer_size, self.input_dim, device)

        self.grad_norm_clip = grad_norm_clip

        if not self.callbacks is None:
            self.callbacks.set_agent(self)

        if not self.logger is None:
            self.logger.log("parameters/learning_rate_initial", self.model.learning_rate.initial_value)
            self.logger.log("parameters/learning_rate_final", self.model.learning_rate.final_value)
            self.logger.log("parameters/learning_rate_decay", self.model.learning_rate.delta)

            self.logger.log("parameters/epsilon_initial", self.epsilon.initial_value)
            self.logger.log("parameters/epsilon_final", self.epsilon.final_value)
            self.logger.log("parameters/epsilon_decay", self.epsilon.delta)

            self.logger.log("parameters/gamma", self.gamma)
            self.logger.log("parameters/batch_size", self.batch_size)
            self.logger.log("parameters/experience_buffer_size", experience_buffer_size)
            self.logger.log("parameters/target_network_sync_freq", self.target_network_sync_freq)
            self.logger.log("parameters/grad_norm_clip", self.grad_norm_clip)

        self.losses = []


    def create_model(self, learning_rate, architecture, device) -> Model:
        return Model(self.input_dim, self.action_dim, learning_rate, architecture, device)

    def getAction(self, state, mask=None, deterministic=False) -> int:
        if mask is None:
            mask = np.ones(self.action_dim, dtype=np.bool)

        if np.random.rand() < self.epsilon.get() and not deterministic:
            prob = np.array(mask, dtype=np.float)
            prob /= np.sum(prob)
            random_action = np.random.choice(self.action_dim, 1, p=prob).item()
            return random_action
        else:
            with th.no_grad():
                mask = np.invert(mask)
                state = th.tensor(state, dtype=th.float).to(self.model.device)
                q_values = self.model.q_values(state)
                q_values[mask] = -th.inf
                return q_values.argmax().item()

    def calculate_loss(self) -> th.Tensor:
        samples = self.exp_buffer.sample(self.batch_size)

        # calculating q-values for states
        states_action_values = self.model.q_values(samples.states).gather(1, samples.actions.unsqueeze(-1)).squeeze(-1)

        # using no grad to avoid updating the target network
        with th.no_grad():

            # getting the maximum q-values for next states (using the target network)
            next_states_values = self.model.q_target(samples.next_states).max(1)[0]

            # Calculating the target values (Q(s_next, a) = 0 if state is terminal)
            expected_state_action_values = samples.rewards + ((~samples.dones) * self.gamma * next_states_values)

        return self.model.loss_func(states_action_values, expected_state_action_values).mean()

    def step(self) -> None:
        if len(self.exp_buffer) < self.batch_size:
            return

        self.model.train(True)
        loss = self.calculate_loss()
        self.model.optimizer.zero_grad()
        
        self.losses.append(loss.item())
        if not self.logger is None and self.num_timesteps % self.log_freq == 0:
            self.logger.log("train/loss", loss.item())

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.model.optimizer.step()


    def update(self, state, action, reward, done, next_state, info) -> None:
        self.exp_buffer.add(state, action, reward, done, next_state)
        self.step()
        self.epsilon.update()
        self.model.update_learning_rate()

        if not self.logger is None and self.num_timesteps % self.log_freq == 0:
            self.logger.log("parameters/learning_rate", self.model.learning_rate.get())
            self.logger.log("parameters/epsilon", self.epsilon.get())

        if self.num_timesteps % self.target_network_sync_freq == 0:
            self.model.sync()

    def print(self) -> None:
        super().print()        # value = round(value, 8)

        print("| {}\t|".format(self.__class__.__name__).expandtabs(45))
        print("| Learning rate: {}\t|".format(self.model.learning_rate.get()).expandtabs(45))
        print("| Epsilon: {}\t|".format(self.epsilon.get()).expandtabs(45))
        print("| Steps until sync: {}\t|".format(self.target_network_sync_freq - (self.num_timesteps % self.target_network_sync_freq)).expandtabs(45))
        if len(self.exp_buffer) > self.batch_size:
            print("| Avg loss: {}\t|".format(np.mean(self.losses[-30:])).expandtabs(45))
        print("|" + "=" * 44 + "|")

    def save(self, file) -> None:
        self.model.save(file)

    def load(self, file) -> None:
        print("Loading from: {}".format(file))
        self.model.load(file)

        try:
            self.exp_buffer = pickle.load(open(self.savedir + "experience_buffer.txt", 'rb'))
        except:
            print("Could not load Experience buffer... reseting experiences")

        try:
            self.loadParameters()
            self.logger.clear()
        except:
            print("Could not load parameters from log... using the ones given in constructor")
            

    def loadParameters(self):
        if not self.logger.load():
            return

        self.model.learning_rate.cur_value = self.logger.data['parameters']['learning_rate']['data'][-1]
        self.model.learning_rate.final_value = self.logger.data['parameters']['learning_rate_final']['data'][-1]
        self.model.learning_rate.delta = self.logger.data['parameters']['learning_rate_decay']['data'][-1]

        self.epsilon.cur_value = self.logger.data['parameters']['epsilon']['data'][-1]
        self.epsilon.final_value = self.logger.data['parameters']['epsilon_final']['data'][-1]
        self.epsilon.delta = self.logger.data['parameters']['epsilon_decay']['data'][-1]

        self.gamma = self.logger.data['parameters']['gamma']['data'][-1]
        self.batch_size = self.logger.data['parameters']['batch_size']['data'][-1]
        self.experience_buffer_sz = self.logger.data['parameters']['experience_buffer_size']['data'][-1]
        self.target_network_sync_freq = self.logger.data['parameters']['target_network_sync_freq']['data'][-1]
        self.grad_norm_clip = self.logger.data['parameters']['grad_norm_clip']['data'][-1]