import torch.nn as nn
from RL_Methods.DQN.Noisy.NoisyLinear import NoisyLinear, NoisyFactorizedLinear
from RL_Methods.DQN.DQNModel import DQNModel

class NoisyModel(DQNModel):

    def __init__(self, input_dim, action_dim, learning_rate, sigma_init, architecture, device) -> None:
        self.sigma_init = sigma_init
        super().__init__(input_dim, action_dim, learning_rate, architecture, device)

    def _make_network(self, achitecture, input_dim, output_dim, use_noise, device) -> nn.Sequential:
        activation = achitecture['activation_fn']
        net_arch = achitecture['net_arch']

        net = nn.Sequential()
        last_dim = input_dim[0]
        for i in range(len(net_arch)):
            net.add_module("layer_{}".format(i+1), NoisyFactorizedLinear(last_dim, net_arch[i], self.sigma_init, True, use_noise))
            net.add_module("activation_{}".format(i+1), activation())
            last_dim = net_arch[i]
        net.add_module("ouput", NoisyFactorizedLinear(last_dim, output_dim, self.sigma_init, bias=True))
        return net.float().to(device)

    def _create_online_network(self, achitecture, input_dim, action_dim, device):
        self.q_net = self._make_network(achitecture, input_dim, action_dim, True, device)

    def _create_target_network(self, achitecture, input_dim, action_dim, device):
        self.target_net = self._make_network(achitecture, input_dim, action_dim, False, device)
