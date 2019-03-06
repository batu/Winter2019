import gym
import os
import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# multiprocess environment
n_cpu = 6
env = SubprocVecEnv([lambda: gym.make('Breakout-ram-v0') for i in range(n_cpu)])
print(env.reset())
log_dir = "tmp/models/"
os.makedirs(log_dir, exist_ok=True)

best_mean_reward, n_steps = -np.inf, 0
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model_atari.pkl')
  n_steps += 1
  return True


model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./TB/atari/", cliprange=0.1)
model.learn(total_timesteps=60000000, callback=callback)
model.save("ppo2_atari_ram")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
