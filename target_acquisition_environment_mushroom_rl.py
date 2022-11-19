from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import Box

from agentgym import AgentGym

class TargetAcquisitionEnvironment(Environment):
    def __init__(self, num_agents, gamma, horizon):

        self.agentGym = AgentGym(num_agents, 1)

        action_space = Box(-self.agentGym.max_omega,self.agentGym.max_omega,(self.agentGym.actions,))

        low_obs, high_obs = self.agentGym.observations

        observation_space = Box(low_obs, high_obs, high_obs.shape)

        mdp_info = MDPInfo(observation_space=observation_space, action_space=action_space, gamma=gamma, horizon=horizon)

        super().__init__(mdp_info)
    
    def reset(self, state=None):
        state = self.agentGym.reset().reshape((-1,))
        return state

    def step(self, action):
        state, reward, done = self.agentGym.step(action)
        state = state.reshape((-1,))
        return state, reward, done, {}

    def render(self):
        return self.agentGym.plot()


