# Reinforcement Learning For Target Acquisition
Agents trained with soft actor-critic RL to acquire targets while simultaneously avoiding adversaries. If you'd like to learn more, check out our poster [here](https://drive.google.com/file/d/1tqGY37brFiUq4xa9Xf6DAxLJRTWdFgfX/view?usp=share_link).

![](https://github.com/ajordan5/RL-Target-Acquisition/blob/master/doc/agent.gif)

## Objective
Given a mobile agent in an unknown environment containing targets and enemies develop a steering policy that enables the agent to maximize the number of targets found while also avoiding enemies

## Input
* Agent State (x, y, heading)
* Measurements (lidar rays detecting both targets and enemies)
* History grid (25x25 grid; 1 = visited 0 = not yet visited)

## Output
* Heading rate
  * Agent travels at a constant forward velocity

### The agent successfully learns to:
* Stay within the bounds
* Steer towards detected targets
* Explore the entire grid space to find all targets
* Steer away from adversaries

![](https://github.com/ajordan5/RL-Target-Acquisition/blob/master/doc/targets.jpg)


