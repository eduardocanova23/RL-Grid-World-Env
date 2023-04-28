# RL-Grid-World-Env
This repository contains several environments organized in different files named as "GridWorldX" where X is the version of the environment. <br>
All of the versions are based on the same blueprint that is a 4x4 grid with two rewards that vanish when the agent collects them and a fixed terminal state.<br>
The versions are the following:<br>
1 - Basic version, only position is accounted in state space <br>
2 - Position and how many collected rewards are accounted <br>
3 - Position and value of each reward in every 4 directions from the agent are accounted <br>
4 - Position and timestamp are accounted <br>
5 - Binary verison of 3: instead of reward value, there is a binary marker that says wether it's a positive reward or a negative reward <br>

