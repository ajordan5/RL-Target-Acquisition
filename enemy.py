import numpy as np

class Enemy:
    def __init__(self, dt, num_enemies):
        self.state = np.zeros((2, num_enemies))
        self.state[1,:] = np.linspace(0,1,num_enemies)
        self.initial = np.copy(self.state)
        self.offsets = np.random.uniform(0, 2*np.pi, num_enemies)
        self.t = 0
        self.dt = dt

    def update(self):
        self.state[0, :] = 0.5*np.cos(0.75*self.t + self.offsets) + 0.5
        # self.state[1, :] = 0.5*np.sin(0.05*self.t + self.offsets) + 0.5
        self.t += self.dt

    def reset(self):
        self.t = 0
        self.state = self.initial

    def plot(self, ax):
        ax.scatter(self.state[0, :], self.state[1, :], marker='d', color="pink")
        