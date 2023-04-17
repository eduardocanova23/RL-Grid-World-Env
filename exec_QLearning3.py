from gridworld3 import GridWorldEnv3
import gym
import time
import os

env = GridWorldEnv3(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
obs = env.reset()

env.train_QLearning(n_episodes=300000)
total_reward = 0
env.reset(exec=True)
done = False
max_steps = 15

while not done and max_steps != 0:
    
    max_steps -= 1
    action = env.argmaxTable() 
    obs, rewards, done, info = env.step(action)
    env.render()
    total_reward += rewards
    print(f'Estado: {obs}')
    print(f'ACAO: {action}')
    print(f'Recompensa: {rewards}')
    time.sleep(1)

print(f'RECOMPENSA TOTAL: {total_reward}')