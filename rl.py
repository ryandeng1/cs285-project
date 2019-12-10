import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.gail import ExpertDataset
from stable_baselines import DQN, HER
from stable_baselines.her import HERGoalEnvWrapper
from multi_env2 import AirTrafficGym


import gym
# import simulation.traf_env

# dataset = ExpertDataset(expert_path='expert_airtraffic.npz', traj_limitation=1, batch_size=128)

env =  HERGoalEnvWrapper(AirTrafficGym(101))#DummyVecEnv([lambda : AirTrafficGym(101)])
model_class = DQN
goal_selection_strategy = 'future'
model = HER('MlpPolicy', env, model_class, policy_kwargs=dict(layers=[512,512]), verbose=1)
# model.pretrain(dataset, n_epochs=100000)
# print('pretrain done')
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
      env.reset()
  print("Observation: ", obs)

print(np.mean(np.array(finish_values)>0))
#plot_values = np.array(plot_values)[:,0,:]
plot_values = np.array(plot_values)
print(plot_values.shape)
plt.plot(plot_values[:,0],plot_values[:,1])
plt.show()
  #env.render()
