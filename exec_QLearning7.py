from gridworld7 import GridWorldEnv7
import gym
import time
import os
import winsound

env = GridWorldEnv7(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
obs = env.reset()

env.train_QLearning(n_episodes=200000)
total_reward = 0
env.reset(exec=True)
done = False
max_steps = 15
winsound.Beep(1000, 1000)


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