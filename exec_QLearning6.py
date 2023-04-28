from gridworld6 import GridWorldEnv6
import gym
import time
import os

env = GridWorldEnv6(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
obs = env.reset()

env.train_QLearning(n_episodes=300000)
total_reward = 0
env.reset(exec=True)
done = False

while not done:
    
    action = env.argmaxTable() 
    obs, rewards, done, info = env.step(action)
    env.render()
    total_reward += rewards
    print(f'Estado: {obs}')
    print(f'ACAO: {action}')
    print(f'Recompensa: {rewards}')
    time.sleep(1)

print(f'RECOMPENSA TOTAL: {total_reward}')