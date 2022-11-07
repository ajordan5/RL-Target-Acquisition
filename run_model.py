
from target_acquisition_environment_mushroom_rl import TargetAcquisitionEnvironment
from mushroom_rl.core import Core, Agent
from torch import nn
import torch
class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Network, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = torch.tanh(self._h1(torch.squeeze(state, 1).float()))
        features2 = torch.tanh(self._h2(features1))
        a = self._h3(features2)

        return a

horizon = 200
gamma = 0.99

environment = TargetAcquisitionEnvironment(1, gamma=0.99, horizon=200)


agent = Agent.load('saved_models/model')
core = Core(agent, environment)
core.evaluate(n_episodes=5, render=True)