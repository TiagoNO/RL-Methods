import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchinfo
import numpy as np

class DistributionalModel(nn.Module):

    def __init__(self, input_dim, action_dim, learning_rate, n_atoms, min_v, max_v, device) -> None:
        super(DistributionalModel, self).__init__()
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        self.input_dim = input_dim
        self.n_atoms = n_atoms
        self.min_v = min_v
        self.max_v = max_v
        self.delta = (self.max_v - self.min_v) / (self.n_atoms - 1)

        arch = {'net_arch':[24, 24], 'activation_fn':nn.ReLU}
        self.q_net = self.make_network(arch, input_dim, action_dim * n_atoms).to(device)
        self.target_net = self.make_network(arch, input_dim, action_dim * n_atoms).to(device)

        self.loss_func = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

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
        net.add_module("ouput", nn.Linear(last_dim, output_dim, bias=True))
        return net

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

    def sync(self):
        print("Sync target network...")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, file):
        th.save(self.q_net.state_dict(), file)

    def load(self, file):
        self.q_net.load_state_dict(th.load(file, map_location=th.device(self.device)))
        self.sync()