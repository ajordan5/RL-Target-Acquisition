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

import os

import matplotlib.pyplot as plt


def run_MC_test(num_mc_runs, horizon, file):
    gamma = 0.99
    mdp = TargetAcquisitionEnvironment(1, gamma, horizon, file)
    agent = Agent.load('saved_models/' + file)
    core = Core(agent, mdp)
    core.evaluate(n_episodes=num_mc_runs, render=False)

def plot_time_data(directory):
    print("loading files from", directory)
    plt.figure()
    all_plot_data = []
    x_data = []
    for filename in os.listdir(directory):
        time_data = []
        f = os.path.join(directory, filename)
        print("loading files from subdirectory", f)
        for file in os.listdir(f):
            full_path = os.path.join(f, file)
            time_data.append(np.genfromtxt(full_path))
        print(np.array(time_data))
        all_plot_data.append(np.array(time_data))
        x_data.append(filename)
    plt.boxplot(all_plot_data)
    plt.xticks(np.arange(len(x_data))+1, x_data)
    plt.legend
    plt.ylabel("Iterations")
    plt.title("Number Of Iterations Until Caught or Boundary Violation")
    plt.show()

def plot_num_targets(directory):
    print("loading files from", directory)
    plt.figure()
    all_plot_data = []
    x_data = []
    for filename in os.listdir(directory):
        time_data = []
        f = os.path.join(directory, filename)
        print("loading files from subdirectory", f)
        average = np.zeros(1000)
        count = 0
        for file in os.listdir(f):
            if file == '100.txt':
                pass
            else:
                full_path = os.path.join(f, file)
                data_in = np.genfromtxt(full_path)
                data_tmp = data_in[-1] * np.ones(1000)
                data_tmp[0:len(data_in)] = data_in
                average += data_tmp
                count += 1
        average /= count
        count = 0
        var = np.zeros(1000)
        for file in os.listdir(f):
            if file == '100.txt':
                pass
            else:
                full_path = os.path.join(f, file)
                data_in = np.genfromtxt(full_path)
                data_tmp = data_in[-1] * np.ones(1000)
                data_tmp[0:len(data_in)] = data_in
                var += (data_tmp - average) ** 2
                count += 1
        var /= (count -1)
        sigma = .1
        upper = average + sigma * np.sqrt(var)
        lower = average - sigma * np.sqrt(var)
        # plt.fill_between(range(1000), lower, upper)
        plt.plot(range(1000), average, label = filename)
        
    plt.title("Total Targets Acquired Versus Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Targets")
    plt.legend()
    plt.show()


    
if __name__ == "__main__":
    # directory = '/home/grant/repos/github/RL-Target-Acquisition/MC_TESTS/enemy_time_to_caught_test'
    # plot_time_data(directory)
    directory = '/home/grant/repos/github/RL-Target-Acquisition/MC_TESTS/enemy_target_test'
    plot_num_targets(directory)
    # run_MC_test(num_mc_runs=101, horizon=1000, file='enemy_no_grid_grant')