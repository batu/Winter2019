import gym
import numpy as np
import random
import retro

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.cmd_util import make_retro_env, make_atari_snapshot_env
from stable_baselines.common.vec_env import SnapshotVecEnv


from rrt_functions import no_embedding, pix2mem, GO_explore_downsampling
from rrt_functions import random_action, learned_policy, chaos_monkey
from rrt_functions import get_random_goal, get_random_goal_pix2mem
from rrt_functions import euclidian_distance
from rrt_functions import run_rrt, run_rrt_rebase, get_score

# Set up variables
#ENV_NAME = 'PongNoFrameskip-v4'
# ENV_NAME = "Breakout-Atari2600"
ENV_NAME = "SuperMarioWorld-Snes"
#ENV_NAME = 'MountainCar-v0'
seed = random.randint(0,1000)
set_global_seeds(seed)
# Hyperparameters
training_steps = 100
embedding_function = GO_explore_downsampling
action_selection_function = chaos_monkey
distance_measure = euclidian_distance
goal_selection_policy = get_random_goal_pix2mem

# Make the environment using StableBaselines.
if "Frameskip" in ENV_NAME:
    env = make_atari_snapshot_env(ENV_NAME, num_env=1, seed=seed)
    unwrapped_env = env.envs[0].unwrapped
    observation_space = env.reset().shape
    action_space = unwrapped_env.action_space
    env.environment_category = "Atari"
elif True:
    env = retro.make(ENV_NAME)
    env = SnapshotVecEnv([lambda: env], training=False)
    unwrapped_env = env.envs[0].unwrapped
    observation_space = env.reset().shape
    action_space = unwrapped_env.action_space
    env.environment_category = "Retro"
else:
    env = gym.make(ENV_NAME)
    env = SnapshotVecEnv([lambda: env])
    observation_space = env.reset().shape
    action_space = env.action_space
    env.environment_category = "Classic"



print("\n########################################################################")
print(f"#\n# RRT Starting on {ENV_NAME}")
print(f"# The seed is {seed}")
print(f"# The action space is: {action_space}")
print(f"# The observation space is:{observation_space}")
print("#" )
print(f"# The embedding function is {embedding_function.__name__}")
print(f"# The action policy is {action_selection_function.__name__}")
print(f"# The distance metric is {distance_measure.__name__}")
print("# ")
print("########################################################################\n")



def main():
    policy_embeddings, pickles = run_rrt_rebase(env, embedding_function, action_selection_function, distance_measure, goal_selection_policy, rendering=True)


def save_pickles(pickles, pickle_count, save_name):
    saved_picked = np.random.choice(pickles, pickle_count)
    np.save(f'SavedStates/{save_name}.npy', saved_picked)
    print("Saving states complete.")

def metric_compare():
    bbox_random = []
    nn_random = []

    bbox_policy = []
    nn_policy = []
    for _ in range(10):
        seed = random.randint(0,1000)
        set_global_seeds(seed)
        policy_embeddings, pickles = run_rrt(env, pix2mem, learned_policy, distance_measure, get_random_goal_pix2mem, verbose=True, rendering=False)
        random_embeddings, pickles = run_rrt(env, pix2mem, random_action, distance_measure, get_random_goal_pix2mem, verbose=True, rendering=False)

        bbox_sum_policy, nuc_norm_policy = get_score(policy_embeddings)
        bbox_policy.append(bbox_sum_policy)
        nn_policy.append(nuc_norm_policy)
        print(f"The bounding box metric result is {bbox_sum_policy[-1]} for learned_policy")
        print(f"The nuclear norm metric result is {nuc_norm_policy[-1]} for learned_policy")
        bbox_sum_random, nuc_norm_random = get_score(random_embeddings)
        bbox_random.append(bbox_sum_random)
        nn_random.append(nuc_norm_random)

        print(f"The bounding box metric result is {bbox_sum_random[-1]} for random_action")
        print(f"The nuclear norm metric result is {nuc_norm_random[-1]} for random_action")

    np.save("Results/bbox_random.npy", np.array(bbox_random))
    np.save("Results/nn_random.npy", np.array(nn_random))
    np.save("Results/bbox_policy.npy", np.array(bbox_policy))
    np.save("Results/nn_policy.npy", np.array(nn_policy))

    print(f"bbox_random mean: {np.mean(bbox_random)}")
    print(f"nn_random mean: {np.mean(nn_random)}")
    print(f"bbox_policy mean: {np.mean(bbox_policy)}")
    print(f"nn_policy mean: {np.mean(nn_policy)}")

    print(f"bbox_random std: {np.std(bbox_random)}")
    print(f"nn_random std: {np.std(nn_random)}")
    print(f"bbox_policy std: {np.std(bbox_policy)}")
    print(f"nn_policy std: {np.std(nn_policy)}")


if __name__== "__main__":
    main()
