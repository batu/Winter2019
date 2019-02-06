import retro, gym
import numpy as np
import os
import tensorflow as tf
import time, random, sys
import cloudpickle
import argparse
from support_utils import save_hyperparameters, parseArguments
import imageio
import numpy as np
import six
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.cmd_util import make_retro_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.atari_wrappers import Pix2Mem
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv, SnapshotVecEnv, DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines import DQN
from stable_baselines import A2C

# BREADCRUMBS_START
NUM_CPU = 1
seed = random.randint(0,1000)
ENV_NAME = "SuperMarioWorld-Snes"
experiment_name = "SmallNetwork"
human_snapshots = False
training_length = 5000000
# BREADCRUMBS_END
set_global_seeds(seed)
REPLAY = False
TB_path = f"Results/Tensorboard/{experiment_name}/"

run_number = 0
try:
    os.mkdir(TB_path[:-1])
except:
    pass

try:
    os.mkdir(f"{TB_path}README")
except:
    pass

models_path = "Results/SavedModels/"

changes = """Setting up the retro experiment pipeline"""
reasoning = """The small network will learn faster with the pix2mem."""
hypothesis = """Better results, more stable training. """


best_mean_reward, n_steps = -np.inf, 0
last_name = "dummy"
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward, last_name
  # Print stats every 1000 calls

  if (n_steps + 1) % 100 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(run_path), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              if os.path.exists(run_path + f'/{last_name}'):
                  os.remove(run_path + f'/{last_name}')
              print("Saving new best model")
              last_name = f"{best_mean_reward:.1f}_{ENV_NAME}_model.pkl"
              _locals['self'].save(run_path + f'/{best_mean_reward:.1f}_{ENV_NAME}_model.pkl')
  n_steps += 1
  return True


if not REPLAY:
    if len(hypothesis) + len(changes) + len(reasoning) < 10:
        print("NOT ENOUGH LOGGING INFO")
        print("Please write more about the changes and reasoning.")
        exit()

    with open(f"{TB_path}/README/README.txt", "w") as readme:
        start_time_ascii = time.asctime(time.localtime(time.time()))
        algorithm = os.path.basename(__file__)[:-2]
        print(f"Experiment start time: {start_time_ascii}", file=readme)
        print(f"\nAlgorithm:\n{algorithm}", file=readme)
        print(f"\nSeed:\n{seed}", file=readme)
        print(f"\nThe Changes:\n{changes}", file=readme)
        print(f"\nReasoning:\n{reasoning}", file=readme)
        print(f"\nHypothesis:\n{hypothesis}", file=readme)
        print(f"\nResults:\n", file=readme)

folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
run_name = f"run{folder_count}"
run_path = f'{TB_path}{run_name}'
os.mkdir(run_path)

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(["retro_train.py"], f"{run_path}/run_summary.txt", experiment_name=experiment_name)

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
print("Training has started!")

# # multiprocess environment


# env = Monitor(env, filename=run_path, allow_early_resets=True)
# env = VecFrameStack(env, n_stack=4)
# BREADCRUMBS_START
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(act_fun=tf.nn.leaky_relu, net_arch=[32, 32, 32])],
                                           feature_extraction="mlp")

env = retro.make(f'{ENV_NAME}')
env = Monitor(env, filename=run_path, allow_early_resets=True)
env = Pix2Mem(env, "Embeddings/mario_1.h5")
env = SnapshotVecEnv([lambda: env])
policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[32, 32, 32])
# env = make_retro_env('SuperMarioWorld-Snes', num_env=1, seed=seed, logdir=run_path)
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=f"{run_path}",
            cliprange=0.1)
# model = A2C(CustomPolicy, env, verbose=1)
model.learn(total_timesteps=training_length, callback=callback)
# BREADCRUMBS_END
print("The training has completed!")
model.save(f"{run_path}/{ENV_NAME}_final.pkl")

obs = env.reset()
while True:
    print("What what")
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        env.reset()
    env.render()
