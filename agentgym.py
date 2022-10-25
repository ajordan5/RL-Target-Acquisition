import numpy as np

class AgentGym:
    def __init__(self, num_agents):
        self.state = np.zeros((3,num_agents))
        self.num_agents = num_agents
        self.speed = 1
        self.dt = 0.1
        self.reward = 0     # int32 reward that increments with each step
        self.done = 0       # int32 flip to one if sim reaches end criteria

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

    def update_reward():
        reward += 1 # TODO

if __name__ == "__main__":
    gym = AgentGym(3)
    ret = gym.step(np.array([1, 2, 3]))
    print(ret)

    gym = AgentGym(1)
    gym.step(np.array([0.3]))
    print(gym.state)