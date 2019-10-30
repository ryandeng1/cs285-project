from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


import gym
import simulation.atc_gym



env =  DummyVecEnv([lambda : simulation.atc_gym.AtcGym()])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000000)
obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  print("Reward: ", rewards)
  #env.render()