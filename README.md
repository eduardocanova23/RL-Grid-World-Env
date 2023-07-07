# RL-Grid-World-Env
This repository contains several environments organized in different files named as "GridWorldX" where X is the version of the environment. <br>
All of the versions are based on the same blueprint that is a 4x4 grid with two rewards that vanish when the agent collects them and a fixed terminal state.<br>
The versions are the following:<br>
1 - Basic version, only position is accounted in state space <br>
2 - Position and how many collected rewards are accounted <br>
3 - Position and value of each reward in every 4 directions from the agent are accounted <br>
4 - Position and timestamp are accounted <br>
5 - Binary verison of 3: instead of reward value, there is a binary marker that says wether it's a positive reward or a negative reward <br>
6 - Same as 1 but the rewards don't vanish when collected <br>
7 - <br>
8 - State is the existence of yellow or blue diamonds in each position. The goal is fixed in (1,3) <br>
9 - State is a mean of all rewards that are in some direction of the agent weighted on the distance to each reward to it <br>
10- Same as 9 but no negative rewards <br>


pygame
Version: 2.3.0 <br>

stable-baselines3
Version: 1.8.0 <br>

gym
Version: 0.26.2 <br>
