import torch as th
import torch.nn as nn
import torch.optim as optim
import torchinfo
from RL_Methods.DQN.Model import Model

class DistributionalModel(Model):

    def __init__(self, input_dim, action_dim, learning_rate, n_atoms, min_v, max_v, architecture, device) -> None:
        self.n_atoms = n_atoms
        self.min_v = min_v
        self.max_v = max_v
        self.delta = (self.max_v - self.min_v) / (self.n_atoms - 1)
        super().__init__(input_dim, action_dim, learning_rate, architecture, device)
        self.register_buffer("support_vector", th.arange(self.min_v, self.max_v + self.delta, self.delta))
        self.softmax = nn.Softmax(dim=2)

    def make_network(self, achitecture, input_dim, output_dim):
        activation = achitecture['activation_fn']
        net_arch = achitecture['net_arch']

        net = nn.Sequential()
        last_dim = input_dim[0]
        for i in range(len(net_arch)):
            net.add_module("layer_{}".format(i+1), nn.Linear(last_dim, net_arch[i], bias=True))
            net.add_module("activation_{}".format(i+1), activation())
            last_dim = net_arch[i]
        net.add_module("ouput", nn.Linear(last_dim, output_dim * self.n_atoms, bias=True))
        return net.to(self.device)

    def __str__(self) -> str:
        return torchinfo.summary(self.q_net, self.input_dim, device=self.device).__str__()

    def forward(self, state):
        batch_sz = state.shape[0]
        return self.q_net(state).view(batch_sz, self.action_dim, self.n_atoms)

    def q_values(self, state):
        values = self(state)
        probs = self.softmax(values)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, values

    def q_target(self, state):
        batch_sz = state.shape[0]
        values = self.target_net(state).view(batch_sz, self.action_dim, self.n_atoms)
        probs = self.softmax(values)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, values
