import torch as th
import torch.nn as nn
import torch.optim as optim

import itertools
import torchinfo

from RL_Methods.utils.Schedule import Schedule
class DuelingModel(nn.Module):

    def __init__(self, input_dim, action_dim, learning_rate : Schedule, architecture, device) -> None:
        super(DuelingModel, self).__init__()
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        self.input_dim = input_dim

        if architecture is None:
            arch = self.set_default_architecture()
        else:
            arch = architecture

        self.features_extractor = self.make_feature_extractor(arch, device)
        self.value_net, self.advantage_net = self.make_network(arch, device)

        self.target_features_extractor = self.make_feature_extractor(arch, device)
        self.target_value_net, self.target_advantage_net = self.make_network(arch, device)

        self.loss_func = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(itertools.chain(self.features_extractor.parameters(), self.advantage_net.parameters(), self.value_net.parameters()), lr=self.learning_rate.get())

    def set_default_architecture(self):
        return {'feature_arch':[128, 128], 'value_arch':[128, 64], 'advantage_arch':[128, 64], 'activation_fn':nn.ReLU}

    def update_learning_rate(self):
        self.learning_rate.update()
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate.get()

    def make_feature_extractor(self, achitecture, device):
        activation = achitecture['activation_fn']
        net_arch = achitecture['feature_arch']

        net = nn.Sequential()
        last_dim = self.input_dim[0]
        for i in range(len(net_arch)):
            net.add_module("layer_{}".format(i+1), nn.Linear(last_dim, net_arch[i], bias=True))
            net.add_module("activation_{}".format(i+1), activation())
            last_dim = net_arch[i]
        return net.to(device)

    def make_network(self, achitecture, device):
        feature_arch = achitecture['feature_arch']
        activation = achitecture['activation_fn']
        value_arch = achitecture['value_arch']
        advantage_arch = achitecture['advantage_arch']

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
        advantage_net.add_module("ouput", nn.Linear(last_dim, self.action_dim, bias=True))

        return value_net.to(device), advantage_net.to(device)

    def forward(self, state):
        return self.q_values(state)

    def q_values(self, state):
        features = self.features_extractor(state)
        advantage = self.advantage_net(features)
        value = self.value_net(features)
        return value + (advantage - th.mean(advantage))

    def q_target(self, state):
        features = self.target_features_extractor(state)
        advantage = self.target_advantage_net(features)
        value = self.target_value_net(features)
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

    def save(self, file):
        th.save(self.features_extractor.state_dict(), file + "_feature_extractor.pt")
        th.save(self.advantage_net.state_dict(), file + "_advantage_net.pt")
        th.save(self.value_net.state_dict(), file + "_value_net.pt")

    def load(self, file):
        # print("Loading from: {}".format(file))
        self.features_extractor.load_state_dict(th.load(file + "_feature_extractor.pt"))
        self.advantage_net.load_state_dict(th.load(file + "_advantage_net.pt"))
        self.value_net.load_state_dict(th.load(file + "_value_net.pt"))
        self.sync()