# Reinforcement Learning For Target Acquisition
Agents trained with soft actor-critic RL to acquire targets while simultaneously avoiding adversaries.

## Objective
Given a mobile agent in an unknown environment containing targets and enemies develop a steering policy that enables the agent to maximize the number of targets found while also avoiding enemies

## Input
* Agent State (x, y, heading)
* Measurements (lidar rays detecting both targets and enemies)
* History grid (25x25 grid; 1 = visited 0 = not yet visited)

## Output
* Heading rate
  * Agent travels at a constant forward velocity

