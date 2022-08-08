import torchinfo
import torch as th
import torch.nn as nn
import torch.optim as optim

from RL_Methods.utils.Schedule import Schedule


class Model(nn.Module):

    def __init__(self, input_dim, action_dim, learning_rate : Schedule, architecture, device) -> None:
        super(Model, self).__init__()
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        self.input_dim = input_dim

        if architecture is None:
            arch = self.set_default_architecture()
        else:
            arch = architecture

        self.q_net = self.make_network(arch, input_dim, action_dim).to(device)
        self.target_net = self.make_network(arch, input_dim, action_dim).to(device)

        self.loss_func = nn.MSELoss(reduction='none')
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate.get())

    def set_default_architecture(self):
        return {'net_arch':[24, 24], 'activation_fn':th.nn.ReLU}

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

    def update_learning_rate(self):
        self.learning_rate.update()
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate.get()

    def __str__(self) -> str:
        return torchinfo.summary(self.q_net, self.input_dim, device=self.device).__str__()

    def q_values(self, state):
        return self.q_net(state)

    def q_target(self, state):
        return self.target_net(state)

    def sync(self):
        print("Sync target network...")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, file):
        th.save(self.q_net.state_dict(), file)

    def load(self, file):
        self.q_net.load_state_dict(th.load(file, map_location=th.device(self.device)))
        self.sync()

    def predict(self, state, deterministic=False, mask=None):
        q_val = self.q_values(state)
        if not mask is None:
            q_val[mask] = -th.inf
        return q_val.argmax().item()