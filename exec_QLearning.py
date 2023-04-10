from gridworld import GridWorldEnv
import gym
import time
import os

env = GridWorldEnv(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
obs = env.reset()

env.train_QLearning(n_episodes=100000)
total_reward = 0
env.reset()
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

# while True:
#     # Take a random action
#     time.sleep(1)
#     action = env_see.action_space.sample()
#     print(action)
#     obs, reward, done, info = env_see.step(action)
    
#     # Render the game
#     env_see.render()
#     if done == True:
#         break

# env.close()

