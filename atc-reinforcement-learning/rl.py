from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C
from stable_baselines import PPO2


import gym
import simulation.atc_gym






num_train_steps = 10000
num_eval_steps = 10000
n_cpu = 8
#env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])


def run():
	# env = SubprocVecEnv([lambda: simulation.atc_gym.AtcGym() for i in range(n_cpu)])
	env =  DummyVecEnv([lambda : simulation.atc_gym.AtcGym()])

	#model = PPO2(MlpPolicy, env)
	model = A2C(MlpPolicy, env, verbose=1, tensorboard_log='./')
	model.learn(total_timesteps=num_train_steps)
	obs = env.reset()
	for i in range(num_eval_steps):
	  action, _states = model.predict(obs)
	  obs, rewards, done, info = env.step(action)
	  print("Reward: ", rewards, "Done: ", done)
	  env.render()

if __name__ == "__main__":
	run()
