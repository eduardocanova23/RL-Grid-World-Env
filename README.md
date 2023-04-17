# RL-Grid-World-Env
This repository contains several environments organized in different files named as "GridWorldX" where X is the version of the environment.
All of the versions are based on the same blueprint that is a 4x4 grid with two rewards that vanish when the agent collects them and a fixed terminal state.
The versions are the following:
1 - Basic version, only position is accounted in state space
2 - Position and how many collected rewards are accounted
3 - Position and value of each reward in every 4 directions from the agent are accounted
4 - Position and timestamp are accounted
5 - Binary verison of 3: instead of reward value, there is a binary marker that says wether it's a positive reward or a negative reward
