import numpy as np
import matplotlib.pyplot as plt

class AgentGym:
    def __init__(self, num_agents):
        self.state = np.zeros((3,num_agents))
        self.num_agents = num_agents
        self.num_targets = 5
        self.targets = np.zeros((2,self.num_targets))
        self.speed = 0.3
        self.dt = 0.1
        self.reward = 0     # int32 reward that increments with each step
        self.done = 0       # int32 flip to one if sim reaches end criteria
        
        # Setup figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-.01, 1.01)
        self.ax.set_ylim(-.01, 1.01)
        self.state_plot = None
        self.target_plot = None

        # Setup sim
        self.init_targets()

    @property
    def theta(self):
        return self.state[2,:]

    @property
    def x(self):
        return self.state[0,:]

    @property
    def y(self):
        return self.state[1,:]

    def step(self, omega):
        self.state += self.dynamics(omega)*self.dt
        self.bounce()
        self.update_reward()
        return (self.state.astype(np.float32),
                np.array(self.reward, np.int32), 
                np.array(self.done, np.int32))
        
    def dynamics(self, omega):
        xdot = self.speed * np.cos(self.theta)
        ydot = self.speed * np.sin(self.theta)
        thetadot = omega
        return np.block([[xdot],
                        [ydot],
                        [thetadot]])

    def bounce(self):
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
        self.reward += 1 # TODO

    def init_targets(self):
        self.targets = np.random.uniform(size=self.targets.shape)
        self.ax.scatter(self.targets[0,:], self.targets[1,:], color='g')

    def plot(self):
        if self.state_plot:
            self.state_plot.remove()
        self.state_plot = self.ax.scatter(self.state[0,:], self.state[1,:], color='k')
        # plt.show()

if __name__ == "__main__":
    # gym = AgentGym(3)
    # ret = gym.step(np.array([1, 2, 3]))
    # print(ret)
    # gym.plot()

    gym = AgentGym(1)
    for i in range(100):
        omega  = np.random.normal(0, 1)
        gym.step(np.array([omega]))
        # print(gym.state)
        gym.plot()
        plt.pause(0.1)