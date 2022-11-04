import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class AgentGym:
    def __init__(self, num_agents):
        self.init_state = np.array([[0.5,0.5,0]]).T
        self.state = np.tile(self.init_state, num_agents)
        self.num_agents = num_agents
        self.num_targets = 5
        self.targets = np.zeros((2,self.num_targets))

        self.target_sense_dist = 0.2
        self.target_sense_azimuth = np.pi/4
        self.sense_num_rays = 10
        self.target_sense_increment = 2*self.target_sense_azimuth/self.sense_num_rays
        self.target_measurements = np.zeros((self.sense_num_rays, num_agents))
        self.num_claimed_targets = 0

        self.speed = 0.3
        self.dt = 0.1
        self.reward = 0     # int32 reward that increments with each step
        self.done = 0       # int32 flip to one if sim reaches end criteria
        self.time_reward = -1
        self.exit_reward = -1000
        self.target_reward = 5000
        self.num_actions = num_agents #number of inputs for agents
        
        # Setup figure
        self.fig, self.ax = plt.subplots()
        self.state_plot = None

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
        combined = np.concatenate((self.state, self.target_measurements), axis=0).reshape(-1,1)
        return combined

    def reset(self, ):
        self.state = np.tile(self.init_state, self.num_agents)
        self.targets = np.zeros((2,self.num_targets))
        self.init_targets()
        self.reward = 0
        self.done = 0
        return self.full_state


    def step(self, omega):
        self.reset_measurements()
        self.propogate_state(omega)
        self.check_bounds_and_angles()
        if not self.done:
            self.search_targets()
        return (self.full_state.astype(np.float32),
                np.array(self.reward, np.int32), 
                np.array(self.done, np.int32))

    def propogate_state(self, omega):
        next = odeint(self.dynamics, self.state.flatten(), [0, self.dt], args=(omega[0], self.speed,))
        self.state[:,0] = next[1,:]
        self.reward += self.time_reward

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
        for agent_i in range(self.num_agents):
            targets_to_delete = []
            for target_i in range(self.targets.shape[1]):
                diff = self.targets[:,target_i] - self.state[0:2, agent_i] 
                dist_to_target = np.linalg.norm(diff)
                angle_to_target = np.arctan2(diff[1], diff[0])
                measurement_angle = self.wrap_angle_pi2pi(angle_to_target - self.state[2, agent_i])
                if(dist_to_target < self.target_sense_dist and abs(measurement_angle) < self.target_sense_azimuth):
                    self.claimed_plot = self.ax.scatter(self.targets[0, target_i], self.targets[1, target_i], color='r')
                    self.add_target_measurement(measurement_angle, dist_to_target, agent_i)

                    if (dist_to_target < self.speed*self.dt):
                        self.reward += self.target_reward
                        targets_to_delete.append(target_i)
            self.targets = np.delete(self.targets, targets_to_delete, axis=1)
                
    def init_targets(self):
        self.ax.clear()
        self.ax.set_xlim(-.01, 1.01)
        self.ax.set_ylim(-.01, 1.01)

        self.targets = np.random.uniform(size=self.targets.shape)
        self.target_plot = self.ax.scatter(self.targets[0,:], self.targets[1,:], color='g')

    def add_target_measurement(self, angle, distance, agent_i):
        ray_i = int(np.round(angle/self.target_sense_increment))
        self.target_measurements[ray_i, agent_i] = distance

    def reset_measurements(self):
        self.target_measurements = np.ones((self.sense_num_rays, self.num_agents))

    def plot(self):
        if self.state_plot:
            self.state_plot.remove()
        self.state_plot = self.ax.scatter(self.x, self.y, color='k')
        # plt.show()

    @staticmethod
    def wrap_angle_pi2pi(angle):

        # Wrap from 0 to 2*pi or -2*pi
        while abs(angle) > 2*np.pi:
            angle -= 2*np.pi * np.sign(angle)

        if abs(angle) > np.pi:
            angle = -(2*np.pi - abs(angle)) * np.sign(angle)

        return angle

if __name__ == "__main__":
    # gym = AgentGym(3)
    # ret = gym.step(np.array([1, 2, 3]))
    # print(ret)
    # gym.plot()

    gym = AgentGym(1)
    # gym.state[2] = 0.5
    done = 0
    while not done:
        omega  = np.random.normal(0, 1)
        ret = gym.step(np.array([omega]))
        done = ret[2]
        print(ret)
        gym.plot()
        plt.pause(0.1)

    gym.reset()
    done = 0
    while not done:
        omega  = np.random.normal(0, 2)
        ret = gym.step(np.array([omega]))
        done = ret[2]
        print(ret)
        gym.plot()
        plt.pause(0.1)