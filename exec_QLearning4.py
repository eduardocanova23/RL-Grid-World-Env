from gridworld4 import GridWorldEnv4
import gym
import time
import os
import winsound

env = GridWorldEnv4(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
obs = env.reset()

env.train_QLearning(n_episodes=500000)
total_reward = 0
env.reset(exec=True)
done = False
winsound.Beep(2000, 1000)

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