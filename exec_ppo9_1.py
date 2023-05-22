from gridworld9_0 import GridWorldEnv9_0
from gridworld9_1 import GridWorldEnv9_1
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os

log_path = os.path.join('Training', 'Logs')
env = GridWorldEnv9_0(render_mode="rgb_array")
#env = DummyVecEnv([lambda: env]) 
model = PPO('MultiInputPolicy',env,verbose=1,tensorboard_log=log_path)
obs = env.reset()

env_see = GridWorldEnv9_1(render_mode="human")
env_see.reset(exec=True)




model.learn(total_timesteps=50000)
done =  False
print(f'Estado: {obs}')
times_to_execute = 10
while times_to_execute > 0:
    obs = env_see.reset(exec=True)
    times_to_execute -= 1
    total_reward = 0
    while not env_see.terminated:
        print(f'shape: {obs["mean_left"].shape}' )
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_see.step(action.astype(int))
        env_see.render()
        total_reward += rewards
        print(f'Estado: {obs}')
        print(f'ACAO: {action.astype(int)}')
        print(f'Recompensa: {rewards}')
        time.sleep(1)
        
    print(f'RECOMPENSA TOTAL: {total_reward}')