from RL_Methods.DuelingDQN.DuelingModel import DuelingModel
from RL_Methods.NoisyNetDQN.NoisyLinear import NoisyLinear, NoisyFactorizedLinear
import torch.nn as nn
import torch as th


class RainbowModel(DuelingModel):

    def __init__(self, input_dim, action_dim, learning_rate, initial_sigma, n_atoms, min_v, max_v, device) -> None:
        self.initial_sigma = initial_sigma
        self.n_atoms = n_atoms
        self.min_v = min_v
        self.max_v = max_v
        self.delta = (self.max_v - self.min_v) / (self.n_atoms - 1)

        super().__init__(input_dim, action_dim, learning_rate, device)
        self.register_buffer("support_vector", th.arange(self.min_v, self.max_v + self.delta, self.delta))
        self.softmax = nn.Softmax(dim=2)
    
    def make_feature_extractor(self, achitecture, device):
        activation = achitecture['activation_fn']
        net_arch = achitecture['feature_arch']

        net = nn.Sequential()
        last_dim = self.input_dim[0]
        for i in range(len(net_arch)):
            net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, net_arch[i], self.initial_sigma, bias=True))
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
            value_net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, value_arch[i], self.initial_sigma, bias=True))
            value_net.add_module("activation_{}".format(i+1), activation())
            last_dim = value_arch[i]
        value_net.add_module("ouput", NoisyFactorizedLinear(last_dim, self.n_atoms, self.initial_sigma, bias=True))

        advantage_net = nn.Sequential()
        last_dim = feature_arch[-1]
        for i in range(len(advantage_arch)):
            advantage_net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, advantage_arch[i], self.initial_sigma, bias=True))
            advantage_net.add_module("activation_{}".format(i+1), activation())
            last_dim = advantage_arch[i]
        advantage_net.add_module("ouput", NoisyFactorizedLinear(last_dim, self.action_dim * self.n_atoms, self.initial_sigma, bias=True))

        return value_net.to(device), advantage_net.to(device)

    def forward(self, state):
        batch_sz = state.shape[0]
        features = self.features_extractor(state)
        advantage_atoms = self.advantage_net(features).view(batch_sz, self.action_dim, self.n_atoms)
        value_atoms = self.value_net(features).view(batch_sz, self.n_atoms)
        return advantage_atoms, value_atoms

    def q_values(self, state):
        batch_sz = state.shape[0]
        advantage_atoms, value_atoms = self(state)
        advantage_atoms = advantage_atoms - th.mean(advantage_atoms, dim=1).unsqueeze(1)
        advantage_atoms[range(batch_sz)] += value_atoms.unsqueeze(1)
        probs = self.softmax(advantage_atoms)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, advantage_atoms

    def q_target(self, state):
        batch_sz = state.shape[0]
        features = self.target_features_extractor(state)
        advantage_atoms = self.target_advantage_net(features).view(batch_sz, self.action_dim, self.n_atoms)
        value_atoms = self.target_value_net(features).view(batch_sz, self.n_atoms)

        advantage_atoms = advantage_atoms - th.mean(advantage_atoms, dim=1).unsqueeze(1)
        advantage_atoms[range(batch_sz)] += value_atoms.unsqueeze(1)
        probs = self.softmax(advantage_atoms)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, advantage_atoms