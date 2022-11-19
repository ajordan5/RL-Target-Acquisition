#example of soft actor critic method from https://github.com/MushroomRL/mushroom-rl/blob/dev/examples/pendulum_sac.py


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger, Agent
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from target_acquisition_environment_mushroom_rl import TargetAcquisitionEnvironment

from tqdm import trange



class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, state_shape, grid_shape, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.num_states = state_shape
        self.num_grids = grid_shape

        # Grid process
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

        # State process
        self._h1 = nn.Linear(state_shape[0]+1, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, 32)

        # Final process
        self._h5 = nn.Linear(64, n_features)
        self._h6 = nn.Linear(n_features, n_features)
        self._h7 = nn.Linear(n_features, n_features)
        self._h8 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h6.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h7.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h8.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):

        vehicle_state = state[:,:self.num_states[0]] 
        vehicle_action = torch.cat((vehicle_state.float(), action.float()), dim=1)
        grid_state = state[:, self.num_states[0]:]
        grid_state = grid_state.reshape((64, 1, 25, 25)).float()

        # Grid processing
        grid_state = self.pool(F.relu(self.conv1(grid_state)))
        grid_state = self.pool(F.relu(self.conv2(grid_state)))
        grid_state = torch.flatten(grid_state, 1) # flatten all dimensions except batch
        grid_state = F.relu(self.fc1(grid_state))
        grid_state = F.relu(self.fc2(grid_state))
        grid_out = self.fc3(grid_state)

        # State Processing
        features1 = F.relu(self._h1(vehicle_action))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        state_out = self._h4(features3)

        combined = torch.cat((grid_out.float(), state_out.float()), dim=1)

        # Combined process
        features1 = F.relu(self._h5(combined))
        features2 = F.relu(self._h6(features1))
        features3 = F.relu(self._h7(features2))
        combined_out = self._h8(features3)

        return torch.squeeze(combined_out)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, state_shape, grid_shape, **kwargs):
        super(ActorNetwork, self).__init__()

        n_output = output_shape[0]

        self.num_states = state_shape
        self.num_grids = grid_shape

        # Grid process
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)

        # State process
        self._h1 = nn.Linear(state_shape[0], n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_features)
        self._h4 = nn.Linear(n_features, 32)

        # Final process
        self._h5 = nn.Linear(64, n_features)
        self._h6 = nn.Linear(n_features, n_features)
        self._h7 = nn.Linear(n_features, n_features)
        self._h8 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h6.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h7.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h8.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        vehicle_state = state[:,:self.num_states[0]].float() 
        grid_state = state[:, self.num_states[0]:]
        grid_state = grid_state.reshape((-1, 1, 25, 25)).float()

        # Grid processing
        grid_state = self.pool(F.relu(self.conv1(grid_state)))
        grid_state = self.pool(F.relu(self.conv2(grid_state)))
        grid_state = torch.flatten(grid_state, 1) # flatten all dimensions except batch
        grid_state = F.relu(self.fc1(grid_state))
        grid_state = F.relu(self.fc2(grid_state))
        grid_out = self.fc3(grid_state)

        # State Processing
        features1 = F.relu(self._h1(vehicle_state))
        features2 = F.relu(self._h2(features1))
        features3 = F.relu(self._h3(features2))
        state_out = self._h4(features3)

        combined = torch.cat((grid_out.float(), state_out.float()), dim=1)

        # Combined process
        features1 = F.relu(self._h5(combined))
        features2 = F.relu(self._h6(features1))
        features3 = F.relu(self._h7(features2))
        combined_out = self._h8(features3)

        return combined_out


def experiment(alg, mdp, n_epochs, n_steps, n_steps_test):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP

    # Settings
    initial_replay_size = 64
    max_replay_size = 50000
    batch_size = 64
    n_features = 64
    warmup_transitions = 100
    tau = 0.005
    lr_alpha = 3e-4

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    # Approximator
    
    actor_input_shape = mdp.info.observation_space.shape
    grid_shape = 625
    state_shape = (actor_input_shape[0] - grid_shape,)
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda,
                            grid_shape=grid_shape, state_shape=state_shape)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda,
                              grid_shape=grid_shape, state_shape=state_shape)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    
    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda,
                         grid_shape=grid_shape, state_shape=state_shape)

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    max_reward = 0
    min_entropy = 10000

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n+1, J=J, R=R, entropy=E)

        file_name = 'saved_models/model{}'.format(n%5)        
        agent.save(file_name,full_save = True)

        if J > max_reward:
            agent.save('saved_models/best_reward'.format(n%5),full_save = True)
            max_reward = J

        if E < min_entropy:
            agent.save('saved_models/best_entropy'.format(n%5),full_save = True)
            min_entropy = E



if __name__ == '__main__':
    alg = SAC
    horizon = 1000
    gamma = 0.99
    mdp = TargetAcquisitionEnvironment(1,.99, 1000)

    # experiment(alg=alg, mdp=mdp, n_epochs=1000, n_steps=5000, n_steps_test=200)


    agent = Agent.load('saved_models/best_reward')
    core = Core(agent, mdp)
    core.evaluate(n_episodes=5, render=True)

    