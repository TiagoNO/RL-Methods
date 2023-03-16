import torch as th
import torch.nn as nn
from RL_Methods.DQN.DQNModel import DQNModel

class DistributionalModel(DQNModel):

    def __init__(self, input_dim, action_dim, learning_rate, n_atoms, min_v, max_v, architecture, device) -> None:
        self.n_atoms = n_atoms
        self.min_v = min_v
        self.max_v = max_v
        self.delta = (self.max_v - self.min_v) / (self.n_atoms - 1)
        super(DistributionalModel, self).__init__(input_dim, action_dim, learning_rate, architecture, device)

    def _make_network(self, achitecture, input_dim, output_dim, device):
        activation = achitecture['activation_fn']
        net_arch = achitecture['net_arch']

        net = nn.Sequential()
        last_dim = input_dim[0]
        for i in range(len(net_arch)):
            net.add_module("layer_{}".format(i+1), nn.Linear(last_dim, net_arch[i], bias=True))
            net.add_module("activation_{}".format(i+1), activation())
            last_dim = net_arch[i]
        net.add_module("ouput", nn.Linear(last_dim, output_dim * self.n_atoms, bias=True))

        self.register_buffer("support_vector", th.arange(self.min_v, self.max_v + self.delta, self.delta))
        self.support_vector = self.support_vector.to(device)
        self.softmax = nn.Softmax(dim=2).to(device)

        return net.float().to(device)

    def forward(self, state):
        batch_sz = state.shape[0]
        return self.q_net(state).view(batch_sz, self.action_dim, self.n_atoms)

    def target_forward(self, state):
        batch_sz = state.shape[0]
        with th.no_grad():
            return self.target_net(state).view(batch_sz, self.action_dim, self.n_atoms)


    def q_values(self, state):
        values = self.forward(state)
        probs = self.softmax(values)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, values

    def q_target(self, state):
        values = self.target_forward(state)
        probs = self.softmax(values)
        q_values = th.mul(probs, self.support_vector).sum(dim=2)
        return q_values, values

    def predict(self, state, deterministic=False, mask=None):
        with th.no_grad():
            q_val, _ = self.q_values(state.unsqueeze(0))
            q_val = q_val.squeeze(0)
            if not mask is None:
                q_val[mask] = -th.inf
            return q_val.argmax().item()