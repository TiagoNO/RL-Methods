import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchinfo

def make_network(achitecture, input_dim, output_dim):
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

class Model(nn.Module):

    def __init__(self, input_dim, action_dim, learning_rate, device) -> None:
        super(Model, self).__init__()
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        self.input_dim = input_dim

        arch = {'net_arch':[256, 256, 256], 'activation_fn':nn.ReLU}
        self.q_net = make_network(arch, input_dim, action_dim).to(device)
        self.target_net = make_network(arch, input_dim, action_dim).to(device)

        self.loss_func = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)

    def __str__(self) -> str:
        return torchinfo.summary(self.q_net, self.action_dim, device=self.device)

    def q_values(self, state):
        return self.q_net(state)

    def q_target(self, state):
        return self.target_net(state)
