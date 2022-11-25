import torch as th
import torch.nn as nn
import torch.optim as optim

import itertools
import torchinfo

from RL_Methods.utils.Schedule import Schedule
from RL_Methods.DQN.DQNModel import DQNModel

class DuelingModel(DQNModel):

    def __init__(self, input_dim, action_dim, learning_rate : Schedule, architecture, device) -> None:
        super(DuelingModel, self).__init__(input_dim, action_dim, learning_rate, architecture, device)

    def _set_optmizer(self):
        self.optimizer = optim.Adam(itertools.chain(self.features_extractor.parameters(), self.advantage_net.parameters(), self.value_net.parameters()), lr=self.learning_rate.get())

    def _create_online_network(self, achitecture, input_dim, action_dim, device):
        self.features_extractor, self.value_net, self.advantage_net = self._make_network(achitecture, input_dim, action_dim, device)

    def _create_target_network(self, achitecture, input_dim, action_dim, device):
        self.target_features_extractor, self.target_value_net, self.target_advantage_net = self._make_network(achitecture, input_dim, action_dim, device)

    def _set_default_architecture(self):
        return {
                'feature_arch':[24, 24], 
                'value_arch':[24], 
                'advantage_arch':[24],
                'activation_fn':th.nn.ReLU
                }

    def _make_network(self, achitecture, input_dim, output_dim, device):
        feature_arch = achitecture['feature_arch']
        activation = achitecture['activation_fn']
        value_arch = achitecture['value_arch']
        advantage_arch = achitecture['advantage_arch']

        feature_extractor = nn.Sequential()
        last_dim = input_dim[0]
        for i in range(len(feature_arch)):
            feature_extractor.add_module("layer_{}".format(i+1), nn.Linear(last_dim, feature_arch[i], bias=True))
            feature_extractor.add_module("activation_{}".format(i+1), activation())
            last_dim = feature_arch[i]

        value_net = nn.Sequential()
        last_dim = feature_arch[-1]
        for i in range(len(value_arch)):
            value_net.add_module("layer_{}".format(i+1), nn.Linear(last_dim, value_arch[i], bias=True))
            value_net.add_module("activation_{}".format(i+1), activation())
            last_dim = value_arch[i]
        value_net.add_module("ouput", nn.Linear(last_dim, 1, bias=True))

        advantage_net = nn.Sequential()
        last_dim = feature_arch[-1]
        for i in range(len(advantage_arch)):
            advantage_net.add_module("layer_{}".format(i+1), nn.Linear(last_dim, advantage_arch[i], bias=True))
            advantage_net.add_module("activation_{}".format(i+1), activation())
            last_dim = advantage_arch[i]
        advantage_net.add_module("ouput", nn.Linear(last_dim, output_dim, bias=True))

        return feature_extractor.float().to(device), value_net.float().to(device), advantage_net.float().to(device)

    def forward(self, state):
        features = self.features_extractor(state)
        advantage = self.advantage_net(features)
        value = self.value_net(features)
        return advantage, value

    def target_forward(self, state):
        with th.no_grad():
            features = self.target_features_extractor(state)
            advantage = self.target_advantage_net(features)
            value = self.target_value_net(features)
            return advantage, value

    def q_values(self, state):
        advantage, value = self.forward(state)
        return value + (advantage - th.mean(advantage))

    def q_target(self, state):
        advantage, value = self.target_forward(state)
        return value + (advantage - th.mean(advantage))

    def __str__(self) -> str:
        features = torchinfo.summary(self.features_extractor, self.input_dim, device=self.device).__str__()
        value = torchinfo.summary(self.value_net, 128, device=self.device).__str__()
        advantage = torchinfo.summary(self.advantage_net, 128, device=self.device).__str__()
        return features + value + advantage

    def sync(self):
        print("Sync target network...")
        self.target_features_extractor.load_state_dict(self.features_extractor.state_dict())
        self.target_advantage_net.load_state_dict(self.advantage_net.state_dict())
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        print(self.target_advantage_net)

    def save(self, file):
        th.save(self.features_extractor.state_dict(), file + "_feature_extractor.pt")
        th.save(self.advantage_net.state_dict(), file + "_advantage_net.pt")
        th.save(self.value_net.state_dict(), file + "_value_net.pt")

    def load(self, file):
        self.features_extractor.load_state_dict(th.load(file + "_feature_extractor.pt", map_location=th.device(self.device)))
        self.advantage_net.load_state_dict(th.load(file + "_advantage_net.pt", map_location=th.device(self.device)))
        self.value_net.load_state_dict(th.load(file + "_value_net.pt", map_location=th.device(self.device)))
        self.sync()