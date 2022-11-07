
from target_acquisition_environment_mushroom_rl import TargetAcquisitionEnvironment
from mushroom_rl.core import Core, Agent

horizon = 200
gamma = 0.99

environment = TargetAcquisitionEnvironment(1, gamma=0.99, horizon=200)


agent = Agent.load('saved_models/model')
core = Core(agent, environment)
core.evaluate(n_episodes=5, render=True)