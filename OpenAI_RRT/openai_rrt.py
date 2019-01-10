import numpy as np
import random

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_dummy_env
from rrt_functions import no_embedding
from rrt_functions import random_action


# Set up variables
ENV_NAME = 'PongNoFrameskip-v4'
seed = 0

# Hyperparameters
training_steps = 100000;
embedding_function = no_embedding
action_selection_function = random_action

# Make the environment using StableBaselines.
env = make_atari_dummy_env(ENV_NAME, num_env=1, seed=seed)
unwrapped_env = env.envs[0].unwrapped

action_space = unwrapped_env.action_space
observation_space = env.reset()[0,:].shape
print("\n########################################################################")
print(f"#\n# RRT Starting on {ENV_NAME}")
print(f"# The action space is: {action_space}")
print(f"# The observation space is:{observation_space}")
print("#" )
print(f"# The embedding function is {embedding_function.__name__}")
print(f"# The action policy is {action_selection_function.__name__}")
print("# ")
print("########################################################################\n")


print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
print(action_selection_function(action_space))
exit()

obs = env.reset()

print(embedding_function("test"))

def main():
    for _ in range(training_steps):
        action = random_action()
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__== "__main__":
    main()
