from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
from stable_baselines import PPO2


import gym
import simulation.atc_gym
import simulation.simulator






num_train_steps = 10000000
num_eval_steps = 100000
n_cpu = 8
#env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
def run():
	sum_rewards = []
	good_reward = False
	# env = SubprocVecEnv([lambda: simulation.atc_gym.AtcGym() for i in range(n_cpu)])
	# env =  DummyVecEnv([lambda : simulation.atc_gym.AtcGym()])
	env =  DummyVecEnv([lambda : simulation.simulator.AirplaneSimulator()])


	# model = PPO2(MlpPolicy, env, verbose=1)
	model = A2C(MlpPolicy, env, verbose=1, tensorboard_log='./')
	model.learn(total_timesteps=num_train_steps)
	obs = env.reset()
	for i in range(num_eval_steps):
	  action, _states = model.predict(obs)
	  obs, rewards, done, info = env.step(action)
	  sum_rewards.append(rewards[0])
	  if done:
	  	obs = env.reset()
	  #print("Reward: ", rewards, "Done: ", done)
	  if rewards[0] != -100:
	  	print("Reward: ", rewards, "Done: ", done)
	  	good_reward = True
	  #env.render()

	if not good_reward:
		print("Boo")

	print("Average reward across {0} steps: {1}".format(num_eval_steps, sum(sum_rewards) / len(sum_rewards)))

if __name__ == "__main__":
	run()
