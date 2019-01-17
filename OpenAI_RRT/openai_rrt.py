import gym
import numpy as np
import random

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_snapshot_env
from stable_baselines.common.vec_env import SnapshotVecEnv
from rrt_functions import no_embedding
from rrt_functions import random_action
from rrt_functions import euclidian_distance
from rrt_functions import run_rrt
from rrt_functions import get_random_goal

# Set up variables
ENV_NAME = 'PongNoFrameskip-v4'
# ENV_NAME = 'MountainCar-v0'
seed = 0

# Hyperparameters
training_steps = 10;
embedding_function = no_embedding
action_selection_function = random_action
distance_measure = euclidian_distance
get_goal = get_random_goal

# Make the environment using StableBaselines.
if "Frameskip" in ENV_NAME:
    env = make_atari_snapshot_env(ENV_NAME, num_env=1, seed=seed)
    unwrapped_env = env.envs[0].unwrapped
    observation_space = env.reset().shape
    action_space = unwrapped_env.action_space
    env.environment_category = "Atari"
else:
    env = gym.make(ENV_NAME)
    env = SnapshotVecEnv([lambda: env])
    observation_space = env.reset().shape
    action_space = env.action_space
    env.environment_category = "Classic"

print("\n########################################################################")
print(f"#\n# RRT Starting on {ENV_NAME}")
print(f"# The action space is: {action_space}")
print(f"# The observation space is:{observation_space}")
print("#" )
print(f"# The embedding function is {embedding_function.__name__}")
print(f"# The action policy is {action_selection_function.__name__}")
print(f"# The distance metric is {distance_measure.__name__}")
print("# ")
print("########################################################################\n")


obs = env.reset()
def main():
    run_rrt(env, embedding_function, action_selection_function, distance_measure, get_goal)

if __name__== "__main__":
    main()
