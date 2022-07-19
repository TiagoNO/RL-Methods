from RL_Methods.DuelingDQN.DuelingModel import DuelingModel
from RL_Methods.NoisyNetDQN.NoisyLinear import NoisyLinear, NoisyFactorizedLinear
import torch.nn as nn

class RainbowModel(DuelingModel):

    def __init__(self, input_dim, action_dim, learning_rate, initial_sigma, device) -> None:
        self.initial_sigma = initial_sigma
        super().__init__(input_dim, action_dim, learning_rate, device)

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
        value_net.add_module("ouput", NoisyFactorizedLinear(last_dim, 1, self.initial_sigma, bias=True))

        advantage_net = nn.Sequential()
        last_dim = feature_arch[-1]
        for i in range(len(advantage_arch)):
            advantage_net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, advantage_arch[i], self.initial_sigma, bias=True))
            advantage_net.add_module("activation_{}".format(i+1), activation())
            last_dim = advantage_arch[i]
        advantage_net.add_module("ouput", NoisyFactorizedLinear(last_dim, self.action_dim, self.initial_sigma, bias=True))

        return value_net.to(device), advantage_net.to(device)