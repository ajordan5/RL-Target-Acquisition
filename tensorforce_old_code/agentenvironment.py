from tensorforce import Environment
from agentgym import AgentGym


class AgentEnvironment(Environment):

    def __init__(self, num_agents):
        super().__init__()
        self.agent = AgentGym(num_agents)

    def states(self):
        return dict(type='float', shape= (self.agent.full_state.shape[0],) )

    def actions(self):
        return dict(type='float', shape = (1,))

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        self.agent.reset()
        return self.agent.full_state.reshape((-1,))

    def execute(self, actions):
        next_state, reward, terminal = self.agent.step(actions)
        next_state = next_state.reshape((-1,))
        return next_state, terminal, reward
    
    def plot(self):
        self.agent.plot()