import retro, gym
import numpy as np
import os
import tensorflow as tf
import time, random, sys
import cloudpickle
import argparse
from support_utils import save_hyperparameters, parseArguments, send_email, visualize_run
import imageio
import numpy as np
import six
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.cmd_util import make_retro_env,  make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.atari_wrappers import Pix2Mem, ClipRewardEnv, WarpFrame, ScaledFloatFrame, EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv
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


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

class PretrainedCnnPolicy(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(PretrainedCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="preloaded_cnn", **_kwargs)


# BREADCRUMBS_START
NUM_CPU = 1
seed = random.randint(0,1000)
ENV_NAME = "BreakoutNoFrameskip-v4"
experiment_name = "test"
state_loading_dict = None
training_length = int(1e7)

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

changes = """I have implemented the pretrained network. 
It is /home/batu/Desktop/HB/FeatureExtractor/breakout_trained_nature_cnn_4_512.h5 It was trained on 10000 steps of breakthrough human play"""

reasoning = """Pre training the feature extractor should make things go faster."""

hypothesis = """ """


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


folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
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
save_hyperparameters(["retro_train.py"], f"{run_path}/run_summary.txt", experiment_name=experiment_name)

start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()

# env = retro.make(f'{ENV_NAME}')
# env = Discretizer(env, combos=[["BUTTON"], ["LEFT"], ["RIGHT"]])
# env = WarpFrame(env)
# env = ScaledFloatFrame(env)
# env = ClipRewardEnv(env)
# env = Monitor(env, filename=run_path, allow_early_resets=True)
# env = SnapshotVecEnv([lambda: env], state_loading_dict=state_loading_dict)
# env = VecFrameStack(env, 4)

# env = make_retro_env(f'{ENV_NAME}', num_env=1, seed=seed, logdir=run_path)
# env = gym.make('Breakout-ram-v0')
# env = retro.make(f'{ENV_NAME}', obs_type=retro.Observations.RAM)
# env = retro.make(f'{ENV_NAME}')
# env = Monitor(env, filename=run_path, allow_early_resets=True)
# env = Pix2Mem(env, "Embeddings/mario_1.h5")
# env = SnapshotVecEnv([lambda: env], state_loading_dict)
env = gym.make(ENV_NAME)
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env = WarpFrame(env)
env = Monitor(env, filename=run_path, allow_early_resets=True)
env = EpisodicLifeEnv(env)
env = ClipRewardEnv(env)
env = ScaledFloatFrame(env)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4)
#env = SubprocVecEnv([lambda:env for i in range(1)])
model = PPO2(PretrainedCnnPolicy, env, verbose=1, tensorboard_log=f"{run_path}",
                    n_steps=128,
                    ent_coef=0.01,
                    learning_rate=linear_schedule(2.5e-4),
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    lam=0.95,
                    nminibatches=4,
                    noptepochs=4,
                    cliprange=linear_schedule(0.1))
print("Training has started!")
model.learn(total_timesteps=training_length, callback=callback)
# BREADCRUMBS_END
print("The training has completed!")
model.save(f"{run_path}/{ENV_NAME}_final.pkl")
visualize_run(run_path)
send_email(f"The {experiment_name}, run {run_number} ended.", run_path)
