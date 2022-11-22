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

from sac import ActorNetwork, CriticNetwork


horizon = 1000
gamma = 0.99
mdp = TargetAcquisitionEnvironment(1,.99, 1000)
agent = Agent.load('saved_models/enemy_no_gird')
core = Core(agent, mdp)
core.evaluate(n_episodes=5, render=True)