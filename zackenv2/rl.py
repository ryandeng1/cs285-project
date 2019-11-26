import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import ExpertDataset
from stable_baselines import PPO2, A2C


import gym
import simulation.traf_env

dataset = ExpertDataset(expert_path='expert_airtraffic.npz', traj_limitation=1, batch_size=128)

env =  DummyVecEnv([lambda : simulation.traf_env.AirTrafficGym()])
model = PPO2(MlpPolicy, env, verbose=1)
model.pretrain(dataset, n_epochs=100000)
print('pretrain done')
model.learn(total_timesteps=100000)
obs = env.reset()
finish_values = []
plot_values = []
for i in range(2000):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  plot_values.append(obs[:2])
  if done:
      print("Reward: ", rewards)
      finish_values.append(rewards)
  print("Observation: ", obs)

print(np.mean(np.array(finish_values)>0))
plot_values = np.array(plot_values)[:,0,:]
print(plot_values.shape)
plt.plot(plot_values[:,0],plot_values[:,1])
plt.show()
  #env.render()
