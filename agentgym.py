import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from enemy import Enemy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mushroom_rl.core import Agent


# class CriticNetwork(nn.Module):
#     def __init__(self, input_shape, output_shape, n_features, state_shape, grid_shape, **kwargs):
#         super().__init__()

#         n_input = input_shape[-1]
#         n_output = output_shape[0]

#         self.num_states = state_shape
#         self.num_grids = grid_shape

#         # Grid process
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(144, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 32)

#         # State process
#         self._h1 = nn.Linear(state_shape[0]+1, n_features)
#         self._h2 = nn.Linear(n_features, n_features)
#         self._h3 = nn.Linear(n_features, n_features)
#         self._h4 = nn.Linear(n_features, 32)

#         # Final process
#         self._h5 = nn.Linear(64, n_features)
#         self._h6 = nn.Linear(n_features, n_features)
#         self._h7 = nn.Linear(n_features, n_features)
#         self._h8 = nn.Linear(n_features, n_output)

#         nn.init.xavier_uniform_(self._h1.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h2.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h3.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h4.weight,
#                                 gain=nn.init.calculate_gain('linear'))
#         nn.init.xavier_uniform_(self._h5.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h6.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h7.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h8.weight,
#                                 gain=nn.init.calculate_gain('linear'))

#     def forward(self, state, action):

#         vehicle_state = state[:,:self.num_states[0]] 
#         vehicle_action = torch.cat((vehicle_state.float(), action.float()), dim=1)
#         grid_state = state[:, self.num_states[0]:]
#         grid_state = grid_state.reshape((64, 1, 25, 25)).float()

#         # Grid processing
#         grid_state = self.pool(F.relu(self.conv1(grid_state)))
#         grid_state = self.pool(F.relu(self.conv2(grid_state)))
#         grid_state = torch.flatten(grid_state, 1) # flatten all dimensions except batch
#         grid_state = F.relu(self.fc1(grid_state))
#         grid_state = F.relu(self.fc2(grid_state))
#         grid_out = self.fc3(grid_state)

#         # State Processing
#         features1 = F.relu(self._h1(vehicle_action))
#         features2 = F.relu(self._h2(features1))
#         features3 = F.relu(self._h3(features2))
#         state_out = self._h4(features3)

#         combined = torch.cat((grid_out.float(), state_out.float()), dim=1)

#         # Combined process
#         features1 = F.relu(self._h5(combined))
#         features2 = F.relu(self._h6(features1))
#         features3 = F.relu(self._h7(features2))
#         combined_out = self._h8(features3)

#         return torch.squeeze(combined_out)


# class ActorNetwork(nn.Module):
#     def __init__(self, input_shape, output_shape, n_features, state_shape, grid_shape, **kwargs):
#         super(ActorNetwork, self).__init__()

#         n_output = output_shape[0]

#         self.num_states = state_shape
#         self.num_grids = grid_shape

#         # Grid process
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(144, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 32)

#         # State process
#         self._h1 = nn.Linear(state_shape[0], n_features)
#         self._h2 = nn.Linear(n_features, n_features)
#         self._h3 = nn.Linear(n_features, n_features)
#         self._h4 = nn.Linear(n_features, 32)

#         # Final process
#         self._h5 = nn.Linear(64, n_features)
#         self._h6 = nn.Linear(n_features, n_features)
#         self._h7 = nn.Linear(n_features, n_features)
#         self._h8 = nn.Linear(n_features, n_output)

#         nn.init.xavier_uniform_(self._h1.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h2.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h3.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h4.weight,
#                                 gain=nn.init.calculate_gain('linear'))
#         nn.init.xavier_uniform_(self._h5.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h6.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h7.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         nn.init.xavier_uniform_(self._h8.weight,
#                                 gain=nn.init.calculate_gain('linear'))

#     def forward(self, state):
#         vehicle_state = state[:,:self.num_states[0]].float() 
#         grid_state = state[:, self.num_states[0]:]
#         grid_state = grid_state.reshape((-1, 1, 25, 25)).float()

#         # Grid processing
#         grid_state = self.pool(F.relu(self.conv1(grid_state)))
#         grid_state = self.pool(F.relu(self.conv2(grid_state)))
#         grid_state = torch.flatten(grid_state, 1) # flatten all dimensions except batch
#         grid_state = F.relu(self.fc1(grid_state))
#         grid_state = F.relu(self.fc2(grid_state))
#         grid_out = self.fc3(grid_state)

#         # State Processing
#         features1 = F.relu(self._h1(vehicle_state))
#         features2 = F.relu(self._h2(features1))
#         features3 = F.relu(self._h3(features2))
#         state_out = self._h4(features3)

#         combined = torch.cat((grid_out.float(), state_out.float()), dim=1)

#         # Combined process
#         features1 = F.relu(self._h5(combined))
#         features2 = F.relu(self._h6(features1))
#         features3 = F.relu(self._h7(features2))
#         combined_out = self._h8(features3)

#         return combined_out

class AgentGym:
    def __init__(self, num_agents=1, num_enemies=0, measure_enemies=True):
        self.init_state = np.array([[0.5,0.5,0]]).T
        self.state = np.tile(self.init_state, num_agents)
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.measure_enemies = measure_enemies
        self.num_targets = 10
        self.targets = np.zeros((2,self.num_targets))

        self.target_sense_dist = 0.2
        self.target_sense_azimuth = np.pi
        self.sense_num_rays = 36
        self.target_sense_increment = 2*self.target_sense_azimuth/(self.sense_num_rays - 1)
        self.target_measurements = np.zeros((self.sense_num_rays, num_agents))        
        self.num_claimed_targets = 0
        self.target_radius = 0.04
        self.claimed = []

        self.speed = 0.3
        self.max_omega = 6
        self.dt = 0.1
        self.reward = 0     # int32 reward that increments with each step
        self.done = 0       # int32 flip to one if sim reaches end criteria
        self.time_reward = -10
        self.exit_reward = -4000
        self.target_reward = 500
        self.num_actions = num_agents #number of inputs for agents

        self.grid_side_length = 25
        self.grid_positions_visited = np.zeros((self.grid_side_length,self.grid_side_length))
        if self.measure_enemies:
            self.enemies = Enemy(self.dt, num_enemies)
            self.enemy_radius = 0.03
            self.caught_reward = -4000
            self.enemy_measurements = np.zeros((self.sense_num_rays, num_agents))
        else:
            self.enemy_measurements = np.array([[]]).T
        
        # Setup figure
        self.save_figs = False
        self.frame_number = 0
        self.fig, self.ax = plt.subplots()
        self.state_plot = None

        #prior
        self.agent = Agent.load('saved_models_grid/best_undiscounted')

        # Setup sim
        self.init_targets()

    @property
    def actions(self):
        return self.num_actions

    @property
    def observations(self): #this function returns np arrays of the min measurment values and the max measurment values
        return np.zeros_like(self.full_state).reshape((-1,)), np.ones_like(self.full_state).reshape((-1,))

    @property
    def theta(self):
        return self.state[2,:]

    @property
    def x(self):
        return self.state[0,:]

    @property
    def y(self):
        return self.state[1,:]

    @property
    def full_state(self):
        # Agent states combined with measurements in one column vector
        
        temp_state = np.concatenate((self.state, self.target_measurements, self.grid_positions_visited.reshape(-1,1)), axis=0).reshape(-1,1)
        action = self.agent.draw_action(temp_state)
        combined = np.concatenate((action, self.enemy_measurements), axis=0).reshape(-1,1)
        return combined

    def reset(self, ):
        self.state = np.tile(self.init_state, self.num_agents)
        self.state[2] = np.random.uniform(-np.pi,np.pi)
        self.targets = np.zeros((2,self.num_targets))
        self.init_targets()
        self.claimed = []
        self.reward = 0
        self.done = 0
        self.grid_positions_visited[:] = 0
        if self.measure_enemies:            
            self.enemies.reset()
        return self.full_state


    def step(self, omega):
        omega = np.clip(omega, -self.max_omega, self.max_omega) 
        # print(omega) 
        self.reward = 0
        self.reset_measurements()
        self.propogate_state(omega)
        self.check_bounds_and_angles()
        if self.measure_enemies:
            self.enemies.update()
            self.search_enemies()
        if not self.done:                
            self.search_targets()
        return (self.full_state.astype(np.float32),
                np.array(self.reward, np.int32), 
                np.array(self.done, np.int32))

    def propogate_state(self, omega):
        next = odeint(self.dynamics, self.state.flatten(), [0, self.dt], args=(omega[0], self.speed,))
        self.state[:,0] = next[1,:]
        # self.reward += self.time_reward

    @staticmethod  
    def dynamics(x, t, omega, speed):
        theta = x[2]
        xdot = speed * np.cos(theta)
        ydot = speed * np.sin(theta)
        thetadot = omega
        return [xdot, ydot, thetadot]

    def check_bounds_and_angles(self):
        # Wrap angles pi to -pi and check if you left the world bounds. Declare done when you leave
        self.state[2,:] = self.wrap_angle_pi2pi(self.state[2,:])
        for i in range(self.num_agents):
            x_i, y_i, theta_i = self.state[:,i]
            
            if(x_i > 1 or x_i < 0 or y_i > 1 or y_i < 0):
                self.done = 1
                self.reward += self.exit_reward
            else:
                self.record_grid_position()

    def record_grid_position(self):
        grid_x_index = int(np.floor(self.x * self.grid_side_length))
        grid_y_index = int(np.floor(self.y * self.grid_side_length))

        
        if self.grid_positions_visited[grid_x_index, grid_y_index] != 1:
            self.grid_positions_visited[grid_x_index, grid_y_index] = 1
            # self.reward += 10
        # else:
            # self.reward -= 10


    def bounce(self):
        # Not currently used. Bounces agents of walls if they leave bounds
        for i in range(self.num_agents):
            x_i, y_i, theta_i = self.state[:,i]
            
            if(x_i > 1 or x_i < 0):
                self.state[2,i] = np.pi - theta_i
                if x_i < 0:
                    self.state[0,i] = 0
                else:
                    self.state[0,i] = 1

            if(y_i > 1 or y_i < 0):
                self.state[2,i] = 2*np.pi - theta_i
                if y_i < 0:
                    self.state[1,i] = 0
                else:
                    self.state[1,i] = 1

    def update_reward(self):
        self.reward -= 1 # TODO

    def search_targets(self):
        # TODO Repeated code with search enemy
        for agent_i in range(self.num_agents):
            targets_to_delete = []
            for target_i in range(self.targets.shape[1]):
                diff = self.targets[:,target_i] - self.state[0:2, agent_i] 
                dist_to_target = np.linalg.norm(diff)
                angle_to_target = np.arctan2(diff[1], diff[0])
                measurement_angle = self.wrap_angle_pi2pi(angle_to_target - self.state[2, agent_i])
                if(dist_to_target < self.target_sense_dist and abs(measurement_angle) < self.target_sense_azimuth):                    
                    self.add_measurement(measurement_angle, dist_to_target, agent_i, self.target_measurements)

                    if (dist_to_target < self.target_radius):
                        self.claimed.append([self.targets[0, target_i], self.targets[1, target_i]])
                        self.reward += self.target_reward
                        targets_to_delete.append(target_i)
                        if len(self.claimed) == self.num_targets:
                            self.done = 1

            self.targets = np.delete(self.targets, targets_to_delete, axis=1)


    def search_enemies(self):
        for agent_i in range(self.num_agents):
            enemy_state = self.enemies.state
            for enemy_i in range(enemy_state.shape[1]):
                diff = enemy_state[:,enemy_i] - self.state[0:2, agent_i] 
                dist_to_target = np.linalg.norm(diff)
                angle_to_target = np.arctan2(diff[1], diff[0])
                measurement_angle = self.wrap_angle_pi2pi(angle_to_target - self.state[2, agent_i])
                if(dist_to_target < self.target_sense_dist and abs(measurement_angle) < self.target_sense_azimuth):                    
                    self.add_measurement(measurement_angle, dist_to_target, agent_i, self.enemy_measurements)

                    if (dist_to_target < self.enemy_radius):
                        self.reward += self.caught_reward
                        self.done = 1
                
    def init_targets(self):
        self.targets = np.random.uniform(size=self.targets.shape, low=0.1, high=0.9)        

    def add_measurement(self, angle, distance, agent_i, measurement_array):
        # Assign measurement to a ray. Rays are shifted to be 2*azimuth to 0.
        # measurements at -azimuth will be index 0 and those at azimuth will be index num_rays - 1
        ray_i = int(np.round((self.target_sense_azimuth+angle)/self.target_sense_increment))
        measurement_array[ray_i, agent_i] = distance

    # def add_enemy_measurement(self, angle, distance, agent_i):
    #     ray_i = int(np.round((self.target_sense_azimuth+angle)/self.target_sense_increment))
    #     self.enemy_measurements[ray_i, agent_i] = distance

    def reset_measurements(self):
        self.target_measurements = np.ones((self.sense_num_rays, self.num_agents))
        if self.measure_enemies:
            self.enemy_measurements = np.ones((self.sense_num_rays, self.num_agents))
    
    def plot_sensor_rays(self):
        ray_off_set = -self.target_sense_azimuth
        for i in range(self.sense_num_rays):
            dist_enemy = 1
            if self.num_enemies > 0:
                dist_enemy = self.enemy_measurements[i]
            dist_target = self.target_measurements[i]
            width_target = 1
            if dist_target == 1:
                dist_target = self.target_sense_dist
                width_target = .1

            angle = self.state[2] + ray_off_set

            x1 = self.state[0]
            y1 = self.state[1]
            if dist_enemy < 1:
                x2_enemy = self.state[0] + dist_enemy * np.cos(angle)
                y2_enemy = self.state[1] + dist_enemy * np.sin(angle)
                width = 1
                self.ax.plot([x1,x2_enemy],[y1,y2_enemy], color = 'red', linewidth = width)
            else:
                x2_target = self.state[0] + dist_target * np.cos(angle)
                y2_target = self.state[1] + dist_target * np.sin(angle)
                self.ax.plot([x1,x2_target],[y1,y2_target], color = 'blue', linewidth = width_target)


            ray_off_set += self.target_sense_increment 

    def plot(self):
        self.ax.clear()
        self.ax.set_xlim(-.01, 1.01)
        self.ax.set_ylim(-.01, 1.01)
        self.ax.set_aspect('equal', 'box')

        self.plot_sensor_rays()
        if self.state_plot:
            self.state_plot.remove()

        if len(self.claimed):
            targets_claimed = np.array(self.claimed)
            self.claimed_plot = self.ax.scatter(targets_claimed[:,0], targets_claimed[:,1], color='r')

        if self.measure_enemies:
            self.enemies.plot(self.ax)
        
        self.state_plot = self.ax.scatter(self.x, self.y, color='k')
        self.target_plot = self.ax.scatter(self.targets[0,:], self.targets[1,:], color='g')
        
        for i in range(self.grid_positions_visited.shape[0]):
            for j in range(self.grid_positions_visited.shape[1]):
                cell = self.grid_positions_visited[i,j]
                if cell == 1:
                    
                    left_bound = i / self.grid_side_length
                    right_bound = (i + 1) / self.grid_side_length
                    bottom_bound = j / self.grid_side_length
                    top_bound = (j + 1) / self.grid_side_length

                    self.ax.fill_between([left_bound, right_bound], [top_bound, top_bound], [bottom_bound, bottom_bound], alpha = .2, color="tab:blue")

        if self.save_figs:
            self.fig.savefig("images/frame{}".format(self.frame_number))
            self.frame_number +=1
        plt.pause(0.02)

    @staticmethod
    def wrap_angle_pi2pi(angle):

        # Wrap from 0 to 2*pi or -2*pi
        while abs(angle) > 2*np.pi:
            angle -= 2*np.pi * np.sign(angle)

        # Now pi to -pi
        if abs(angle) > np.pi:
            angle = -(2*np.pi - abs(angle)) * np.sign(angle)

        return angle

if __name__ == "__main__":
    # gym = AgentGym(3)
    # ret = gym.step(np.array([1, 2, 3]))
    # print(ret)
    # gym.plot()

    gym = AgentGym(1, 0)
    # gym.state[2] = 0.5
    done = 0
    while not done:
        omega  = np.random.normal(0, 1)
        ret = gym.step(np.array([omega]))
        done = ret[2]
        print(ret)
        gym.plot()
        # plt.pause(0.1)

    gym.reset()
    done = 0
    while not done:
        omega  = np.random.normal(0, 2)
        ret = gym.step(np.array([omega]))
        done = ret[2]
        print(ret)
        gym.plot()
        # plt.pause(0.1)