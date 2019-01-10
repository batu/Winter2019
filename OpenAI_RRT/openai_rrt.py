import gym
import numpy as np
import random

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_snapshot_env
from stable_baselines.common.vec_env import SnapshotVecEnv
from rrt_functions import no_embedding
from rrt_functions import random_action


# Set up variables
ENV_NAME = 'PongNoFrameskip-v4'
ENV_NAME = 'MountainCar-v0'
seed = 0

# Hyperparameters
training_steps = 100000;
embedding_function = no_embedding
action_selection_function = random_action

# Make the environment using StableBaselines.
if "FrameSkip" in ENV_NAME:
    env = make_atari_snapshot_env(ENV_NAME, num_env=1, seed=seed)
    unwrapped_env = env.envs[0].unwrapped

    observation_space = env.reset().shape
    action_space = unwrapped_env.action_space
else:
    env = gym.make(ENV_NAME)
    env = SnapshotVecEnv([lambda: env])
    observation_space = env.reset().shape
    action_space = env.action_space


print("\n########################################################################")
print(f"#\n# RRT Starting on {ENV_NAME}")
print(f"# The action space is: {action_space}")
print(f"# The observation space is:{observation_space}")
print("#" )
print(f"# The embedding function is {embedding_function.__name__}")
print(f"# The action policy is {action_selection_function.__name__}")
print("# ")
print("########################################################################\n")

obs = env.reset()
print("Initial state: ", env.envs[0].unwrapped.state)
def main():
    for _ in range(50):
        obs, rewards, dones, info = env.step([0])
        # env.render()
    env.close()
    saved_state = env.get_env_pickle()
    for _ in range(50):
        obs, rewards, dones, info = env.step([2])
        env.render()
    print("Right before loading state: ", env.envs[0].unwrapped.state)
    env.load_env_pickle(saved_state)
    print("Loaded state: ", env.envs[0].unwrapped.state)
    env.render()
    while True:
        pass

if __name__== "__main__":
    main()
