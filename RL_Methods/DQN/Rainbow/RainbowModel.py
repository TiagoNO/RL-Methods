import torch as th
import torch.nn as nn

from RL_Methods.DQN.Dueling.DuelingModel import DuelingModel
from RL_Methods.DQN.Noisy.NoisyLinear import NoisyLinear, NoisyFactorizedLinear


class RainbowModel(DuelingModel):

    def __init__(self, input_dim, action_dim, learning_rate, initial_sigma, n_atoms, min_v, max_v, architecture, device) -> None:
        self.initial_sigma = initial_sigma
        self.n_atoms = n_atoms
        self.min_v = min_v
        self.max_v = max_v
        self.delta = (self.max_v - self.min_v) / (self.n_atoms - 1)

        super().__init__(input_dim, action_dim, learning_rate, architecture, device)

    def _make_network(self, achitecture, input_dim, output_dim, use_noise, device):
        feature_arch = achitecture['feature_arch']
        activation = achitecture['activation_fn']
        value_arch = achitecture['value_arch']
        advantage_arch = achitecture['advantage_arch']

        feature_extractor = nn.Sequential()
        last_dim = input_dim[0]
        for i in range(len(feature_arch)):
            feature_extractor.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, feature_arch[i], self.initial_sigma, bias=True, use_noise=use_noise))
            feature_extractor.add_module("activation_{}".format(i+1), activation())
            last_dim = feature_arch[i]

        value_net = nn.Sequential()
        last_dim = feature_arch[-1]
        for i in range(len(value_arch)):
            value_net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, value_arch[i], self.initial_sigma, bias=True, use_noise=use_noise))
            value_net.add_module("activation_{}".format(i+1), activation())
            last_dim = value_arch[i]
        value_net.add_module("ouput", NoisyFactorizedLinear(last_dim, self.n_atoms, self.initial_sigma, bias=True, use_noise=use_noise))

        advantage_net = nn.Sequential()
        last_dim = feature_arch[-1]
        for i in range(len(advantage_arch)):
            advantage_net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, advantage_arch[i], self.initial_sigma, bias=True, use_noise=use_noise))
            advantage_net.add_module("activation_{}".format(i+1), activation())
            last_dim = advantage_arch[i]
        advantage_net.add_module("ouput", NoisyFactorizedLinear(last_dim, output_dim * self.n_atoms, self.initial_sigma, bias=True, use_noise=use_noise))

        self.register_buffer("support_vector", th.arange(self.min_v, self.max_v + self.delta, self.delta))
        self.support_vector = self.support_vector.to(device)
        self.softmax = nn.Softmax(dim=2).to(device)

        return feature_extractor.float().to(device), value_net.float().to(device), advantage_net.float().to(device)

    def _create_online_network(self, achitecture, input_dim, action_dim, device):
        self.features_extractor, self.value_net, self.advantage_net = self._make_network(achitecture, input_dim, action_dim, True, device)

    def _create_target_network(self, achitecture, input_dim, action_dim, device):
        self.target_features_extractor, self.target_value_net, self.target_advantage_net = self._make_network(achitecture, input_dim, action_dim, False, device)


    def forward(self, state):
        batch_sz = state.shape[0]
        features = self.features_extractor(state)
        advantage_atoms = self.advantage_net(features).view(batch_sz, self.action_dim, self.n_atoms)
        value_atoms = self.value_net(features).view(batch_sz, self.n_atoms)
        return advantage_atoms, value_atoms

    def target_forward(self, state):
        batch_sz = state.shape[0]
        with th.no_grad():
            features = self.target_features_extractor(state)
            advantage_atoms = self.target_advantage_net(features).view(batch_sz, self.action_dim, self.n_atoms)
            value_atoms = self.target_value_net(features).view(batch_sz, self.n_atoms)
            return advantage_atoms, value_atoms


    def q_values(self, state):
        batch_sz = state.shape[0]
        advantage_atoms, value_atoms = self.forward(state)
        
        advantage_atoms = advantage_atoms - th.mean(advantage_atoms, dim=1).unsqueeze(1)
        advantage_atoms[range(batch_sz)] += value_atoms.unsqueeze(1)
        probs = self.softmax(advantage_atoms)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, advantage_atoms

    def q_target(self, state):
        batch_sz = state.shape[0]
        advantage_atoms, value_atoms = self.target_forward(state)

        advantage_atoms = advantage_atoms - th.mean(advantage_atoms, dim=1).unsqueeze(1)
        advantage_atoms[range(batch_sz)] += value_atoms.unsqueeze(1)
        probs = self.softmax(advantage_atoms)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, advantage_atoms

    def predict(self, state, deterministic=False, mask=None):
        with th.no_grad():
            q_val, _ = self.q_values(state.unsqueeze(0))
            q_val = q_val.squeeze(0)
            if not mask is None:
                q_val[mask] = -th.inf
            return q_val.argmax().item()