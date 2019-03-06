import retro
import gym
import numpy as np
import os
import tensorflow as tf
import time
import random
import sys
import cloudpickle
import argparse
from support_utils import save_hyperparameters, parseArguments, send_email, visualize_run
import imageio
import numpy as np
import six
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.cmd_util import make_retro_env,  make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines.common.atari_wrappers import Pix2Mem, ScaledFloatFrame, PixAB2C, WarpFrame
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv, SnapshotVecEnv, DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines import DQN
from stable_baselines import A2C

def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    def func(progress):
        return progress * initial_value
    return func


# BREADCRUMBS_START
NUM_CPU = 1
seed = random.randint(0, 1000)
# ENV_NAME = "SuperMarioWorld-Snes"
ENV_NAME = "BreakoutNoFrameskip-v4"
experiment_name = "breakout_ab2cdelta_8192"
# state_loading_dict = {"load_path":"States/SMW/smw_12_300.pickle",
# state_loading_dict = {"load_path": "States/Breakout/breakout_250.pickle",
#                       "policy": "random"}
state_loading_dict = None

training_length = int(1e7)
test_count = 10
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
episode_counter = 0
external_best = -np.inf
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, last_name, episode_counter, external_best

    # # Print stats every 1000 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(run_path), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                if os.path.exists(run_path + f'/{last_name}'):
                    os.remove(run_path + f'/{last_name}')
                print("Saving new best model")
                last_name = f"{best_mean_reward:.1f}_{ENV_NAME}_model.pkl"
                _locals['self'].save(
                    run_path + f'/{best_mean_reward:.1f}_{ENV_NAME}_model.pkl')
    n_steps += 1

    return True


class PretrainedCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)

folder_count = len([f for f in os.listdir(TB_path)
                    if not os.path.isfile(os.path.join(models_path, f))])
run_name = f"run{folder_count}"
run_path = f'{TB_path}{run_name}'
os.mkdir(run_path)

with open(os.path.join(f"{run_path}", "README.txt"), "w") as readme:
    start_time_ascii = time.asctime(time.localtime(time.time()))
    algorithm = os.path.basename(__file__)[:-2]
    print(f"Experiment start time: {start_time_ascii}", file=readme)
    print(f"\nAlgorithm:\n{algorithm}", file=readme)
    print(f"\nSeed:\n{seed}", file=readme)
    print(f"\nThe Changes:\n{changes}", file=readme)
    print(f"\nReasoning:\n{reasoning}", file=readme)
    print(f"\nHypothesis:\n{hypothesis}", file=readme)
    print(f"\nResults:\n", file=readme)

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(
    ["retro_train.py"], f"{run_path}/run_summary.txt", experiment_name=experiment_name)

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
print("Training has started!")

# env = Monitor(env, filename=run_path, allow_early_resets=True)
# BREADCRUMBS_START


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[
                                               dict(act_fun=tf.nn.leaky_relu, net_arch=[1, 32, 32, 32])],
                                           feature_extraction="mlp")


# env = retro.make(f'{ENV_NAME}')
# env = ScaledFloatFrame(env)
# env = Pix2Mem(env, "Embeddings/breakout_encoder.h5")
# env = Monitor(env, filename=run_path, allow_early_resets=True)
# env = SnapshotVecEnv([lambda: env], state_loading_dict=state_loading_dict)
env = gym.make(f"{ENV_NAME}")
# env = WarpFrame(env)
env = ScaledFloatFrame(env)
env = PixAB2C(env, "Embeddings/breakout_ab2cdelta_8192.h5")
env = Monitor(env, filename=run_path, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4)
#env = SubprocVecEnv([lambda:env for i in range(1)])
model = PPO2(MlpPolicy, env, verbose=1, #tensorboard_log=f"{run_path}",
                    n_steps=128,
                    ent_coef=0.01,
                    learning_rate=linear_schedule(2.5e-4),
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    lam=0.95,
                    nminibatches=4,
                    noptepochs=4,
                    cliprange=linear_schedule(0.1))

model.learn(total_timesteps=training_length, callback=callback)
# current_training_length = 0
# while training_length > current_training_length:
#     train_step = int(training_length / test_count)
#     current_training_length += train_step
#     model.learn(total_timesteps=train_step, callback=callback)
#     print(f"The reward in test {int(current_training_length / train_step)} is:", test_model())
# BREADCRUMBS_END
print("The training has completed!")
model.save(f"{run_path}/{ENV_NAME}_final.pkl")
visualize_run(run_path)
send_email(f"The {experiment_name}, run {run_number} ended.", run_path)
