import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchinfo
from RL_Methods.NoisyNetDQN.NoisyLinear import NoisyLinear, NoisyFactorizedLinear

class NoisyModel(nn.Module):

    def __init__(self, input_dim, action_dim, learning_rate, sigma_init, device) -> None:
        super(NoisyModel, self).__init__()
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = device
        self.input_dim = input_dim
        self.sigma_init = sigma_init

        arch = {'net_arch':[24, 24], 'activation_fn':nn.ReLU}
        self.q_net = self.make_network(arch, input_dim, action_dim).to(device)

        self.target_net = self.make_network(arch, input_dim, action_dim).to(device)

        self.loss_func = nn.MSELoss(reduction='none')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def make_network(self, achitecture, input_dim, output_dim) -> nn.Sequential:
        activation = achitecture['activation_fn']
        net_arch = achitecture['net_arch']

        net = nn.Sequential()
        last_dim = input_dim[0]
        for i in range(len(net_arch)):
            net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, net_arch[i], self.sigma_init, bias=True))
            net.add_module("activation_{}".format(i+1), activation())
            last_dim = net_arch[i]
        net.add_module("ouput", NoisyFactorizedLinear(last_dim, output_dim, self.sigma_init, bias=True))
        return net

    def reset_noise(self):
        # for m in self.q_net.modules():
        #     if isinstance(m, NoisyLinear):
        #         m.reset_noise()
        pass

    def __str__(self) -> str:
        return torchinfo.summary(self.q_net, self.input_dim, device=self.device).__str__()

    def q_values(self, state):
        return self.q_net(state)

    def q_target(self, state):
        return self.target_net(state)

    def sync(self):
        print("Sync target network...")
        self.target_net.load_state_dict(self.q_net.state_dict())