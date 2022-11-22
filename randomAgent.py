from agentgym import AgentGym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange


def random_agent():
    gym = AgentGym(1, num_enemies=0, file="random_with_bounce")
    horizon = 1000
    # gym.state[2] = 0.5
    done = 0
    for i in trange(101):
        current_step = 0
        while not done and current_step < horizon:
            omega  = np.random.normal(-gym.max_omega, gym.max_omega)
            ret = gym.step(np.array([omega]))
            done = ret[2]
            # gym.plot()
            # plt.pause(0.1)
            current_step += 1
        gym.reset()
        done = 0


if __name__ == "__main__":
    random_agent()