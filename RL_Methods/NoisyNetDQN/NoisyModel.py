import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchinfo
from RL_Methods.NoisyNetDQN.NoisyLinear import NoisyLinear, NoisyFactorizedLinear
from RL_Methods.DQN.Model import Model

class NoisyModel(Model):

    def __init__(self, input_dim, action_dim, learning_rate, sigma_init, architecture, device) -> None:
        self.sigma_init = sigma_init
        super().__init__(input_dim, action_dim, learning_rate, architecture, device)

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