## Adapted from example from https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_a2c.py




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import trange

from mushroom_rl.core import Core, Logger
from mushroom_rl.algorithms.actor_critic import A2C

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.core import Agent

from target_acquisition_environment_mushroom_rl import TargetAcquisitionEnvironment


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


def experiment(alg, environment, n_epochs, n_steps, n_steps_per_fit, n_step_test, alg_params, policy_params):
    logger = Logger(A2C.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + A2C.__name__)


    critic_params = dict(network=Network,
                         optimizer={'class': optim.RMSprop,
                                    'params': {'lr': 7e-4,
                                               'eps': 1e-5}},
                         use_cuda=True,
                         loss=F.mse_loss,
                         n_features=64,
                         batch_size=64,
                         input_shape=environment.info.observation_space.shape,
                         output_shape=(1,))

    alg_params['critic_params'] = critic_params


    policy = GaussianTorchPolicy(Network,
                                 environment.info.observation_space.shape,
                                 environment.info.action_space.shape,
                                 **policy_params)

    agent = alg(environment.info, policy, **alg_params)

    core = Core(agent, environment)

    dataset = core.evaluate(n_steps=n_step_test, render=False)

    J = np.mean(compute_J(dataset, environment.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy()

    logger.epoch_info(0, J=J, R=R, entropy=E)

    for it in trange(n_epochs):
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit)
        dataset = core.evaluate(n_steps=n_step_test, render=False)
        if it % 2 == 0:
            agent.save('saved_models/model',full_save = True)


        J = np.mean(compute_J(dataset, environment.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy()

        logger.epoch_info(it+1, J=J, R=R, entropy=E)

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)
    agent.save('saved_models/model',full_save = True)

def loadModel(path):
    return Agent.load(path)


if __name__ == '__main__':
    policy_params = dict(
        std_0=1.,
        n_features=64,
        use_cuda=True
    )

    a2c_params = dict(actor_optimizer={'class': optim.RMSprop,
                                       'params': {'lr': 7e-4,
                                                  'eps': 3e-3}},
                      max_grad_norm=0.5,
                      ent_coeff=0.01)

    algs_params = [
        (A2C, 'a2c', a2c_params)
     ]
     
    env_id='Pendulum-v1'
    horizon = 200
    gamma = 0.99

    environment = TargetAcquisitionEnvironment(1, gamma=0.99, horizon=200)

    for alg, alg_name, params in algs_params:
        experiment(alg=alg, environment=environment,
                   n_epochs=100, n_steps=30000, n_steps_per_fit=5,
                   n_step_test=5000, alg_params=params,
                   policy_params=policy_params)

    # agent = loadModel('saved_models/model')
    # core = Core(agent, environment)
    # dataset = core.evaluate(n_episodes=1, render=True)
    # J = np.mean(compute_J(dataset, environment.info.gamma))
    # R = np.mean(compute_J(dataset))
    # E = agent.policy.entropy()

    # print(J, R, E)