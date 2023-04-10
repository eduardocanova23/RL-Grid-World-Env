from gridworld import GridWorldEnv
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

log_path = os.path.join('Training', 'Logs')
env = GridWorldEnv(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
model = PPO('MultiInputPolicy',env,verbose=1,tensorboard_log=log_path)
obs = env.reset()

env_see = GridWorldEnv(render_mode="human")
env_see.reset()

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

env_see.reset()
model.learn(total_timesteps=20000)
done =  False
print(f'Estado: {obs}')
total_reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env_see.step(action.astype(int))
    env_see.render()
    total_reward += rewards
    print(f'Estado: {obs}')
    print(f'ACAO: {action.astype(int)}')
    print(f'Recompensa: {rewards}')
    time.sleep(1)

print(f'RECOMPENSA TOTAL: {total_reward}')