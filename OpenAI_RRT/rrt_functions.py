import random
import numpy as np

from scipy.misc import imsave
from scipy.spatial import ConvexHull

from skimage.measure import block_reduce
from keras.models import load_model
from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import Pix2Mem
from stable_baselines.common.cmd_util import make_retro_env
from stable_baselines.common.vec_env import VecFrameStack

import sys, os


# This is the main function that starts the rrt search.
def run_rrt(env, embedding_function, action_policy, distance_measure, get_goal,
action_steps=200, max_steps = 1000, verbose=False, rendering=True, state_pruning=False, state_visitation_weight=2):

    #    ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
    print(env.envs[0].unwrapped.buttons)
    # state_visitation_weight default = 2
    # The higher, the more likely it is to pick un sampled places

    # Extracts information about the environment
    unwrapped_env = env.envs[0].unwrapped

    # This can pixel input. This can be RAM, this can be anything
    observation_space = env.reset().shape

    # [1.1.0.0.0.1.0.1.0.1]
    # Action space looks like this. Digits map to a button on a controller. 1 means pressed 0 means not pressed.
    action_space = unwrapped_env.action_space

    embedding_model = None

    action_policy_setup, action_policy_act = action_policy()

    info_dict = action_policy_setup(unwrapped_env)

    # This loads the embedding if necesary
    if embedding_function.__name__ == "pix2mem":
        embedding_model = load_model("Model/smw_pix2mem.h5") 

    # assert type(env) == SnapshotVecEnv

    # obs is of shape (1, 224, 256, 3)
    obs = env.reset()

    # Keep track of the states/embeddings/pickles
    observations = [obs]
    embeddings = [embedding_function(obs, embedding_model)]
    pickles = [env.get_env_pickle()]
    explore_count = [0]

    max_embedding_val = np.array(embeddings[0])
    min_embedding_val = np.array(embeddings[0])


    for i in range(max_steps):

        # print(f"The max val is: {max_obs_val} and the min val is {min_obs_val}")

        # Get a goal
        goal = get_goal(embedding_function(obs, embedding_model), max_embedding_val, min_embedding_val)

        # Find which state is closest to your goal
        chosen_index = np.argmin([distance_measure(em_obs, goal) + (explore_count[index])**state_visitation_weight for index, em_obs in enumerate(embeddings)])
        # chosen_index = np.argmin([(explore_count[index])**state_visitation_weight for index, em_obs in enumerate(embeddings)])
        
        info_dict["chosen_index"] = chosen_index 

        # Count how many times you chose that state. This helps reduce the duplicate choices.
        explore_count[chosen_index] += 1
        if verbose or (i % 100 == 0):
            print(f"In step {i} the chosen index is: {chosen_index}")

        # Load the closest goal.
        chosen_pickle = pickles[chosen_index]
        env.load_env_pickle(chosen_pickle)

        # Get the correct embedding
        chosen_observation = embedding_function(observations[chosen_index], embedding_model)

        # Get the action to do from that chosen goal state.
        # chosen_action = [action_policy_act(chosen_observation, info_dict)]

        # Take that action action step times (usually 30)
        for _ in range(random.randint(0, action_steps)):
            chosen_action = [action_policy_act(chosen_observation, info_dict)] 
            for _ in range(4):
                obs, rewards, dones, info = env.step(chosen_action)
        

        if(rendering and i % 1 == 0):
            env.render()

        # Added the new state to your list.
        explore_count.append(1)
        observations.append(obs)
        embedded_obs = embedding_function(obs, embedding_model)
        embeddings.append(embedded_obs)
        pickles.append(env.get_env_pickle())

        np.maximum(max_embedding_val, embedded_obs, out=max_embedding_val)
        np.minimum(min_embedding_val, embedded_obs, out=min_embedding_val)

        print(f"{sum(max_embedding_val):.3f}, {sum(min_embedding_val):.3f}")

        # Test it fit a gaussian fitted sampling.
        # Test it with mixture gaussians

    print(explore_count)
    print(    max(explore_count))
    print(    min(explore_count))
    print(len(explore_count))
    env.unwrapped.close()
    return embeddings, pickles


# This is the main function that starts the rrt search.
def run_rrt_rebase(env, embedding_function, action_policy, distance_measure, get_goal,
action_steps=200, max_steps=10, rebase_count=10, verbose=False, rendering=True, state_pruning=False, state_visitation_weight=2):

    #    ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
    print(env.envs[0].unwrapped.buttons)
    # state_visitation_weight default = 2
    # The higher, the more likely it is to pick un sampled places

    # Extracts information about the environment
    unwrapped_env = env.envs[0].unwrapped

    # This can pixel input. This can be RAM, this can be anything
    observation_space = env.reset().shape

    # [1.1.0.0.0.1.0.1.0.1]
    # Action space looks like this. Digits map to a button on a controller. 1 means pressed 0 means not pressed.
    action_space = unwrapped_env.action_space

    embedding_model = None

    action_policy_setup, action_policy_act = chaos_monkey()

    info_dict = action_policy_setup(unwrapped_env)

    # This loads the embedding if necesary
    if embedding_function.__name__ == "pix2mem":
        embedding_model = load_model("Model/smw_pix2mem.h5") 

    # assert type(env) == SnapshotVecEnv

    # obs is of shape (1, 224, 256, 3)
    obs = env.reset()

    # Keep track of the states/embeddings/pickles
    observations = [obs]
    embeddings = [embedding_function(obs, embedding_model)]
    pickles = [env.get_env_pickle()]
    explore_count = [0]

    max_embedding_val = np.array(embeddings[0])
    min_embedding_val = np.array(embeddings[0])

    farthest_points = []
    for _ in range(rebase_count):
        for i in range(max_steps):

            # print(f"The max val is: {max_obs_val} and the min val is {min_obs_val}")

            # Get a goal
            goal = get_goal(embedding_function(obs, embedding_model), max_embedding_val, min_embedding_val)

            # Find which state is closest to your goal
            chosen_index = np.argmin([distance_measure(em_obs, goal) + (explore_count[index])**state_visitation_weight for index, em_obs in enumerate(embeddings)])
            # chosen_index = np.argmin([(explore_count[index])**state_visitation_weight for index, em_obs in enumerate(embeddings)])
            
            info_dict["chosen_index"] = chosen_index 

            # Count how many times you chose that state. This helps reduce the duplicate choices.
            explore_count[chosen_index] += 1
            if verbose or (i % 100 == 0):
                print(f"In step {i} the chosen index is: {chosen_index}")

            # Load the closest goal.
            chosen_pickle = pickles[chosen_index]
            env.load_env_pickle(chosen_pickle)

            # Get the correct embedding
            chosen_observation = embedding_function(observations[chosen_index], embedding_model)

            # Take that action action step times (usually 30)
            for _ in range(random.randint(0, action_steps)):
                chosen_action = [action_policy_act(chosen_observation, info_dict)] 
                for _ in range(4):
                    obs, rewards, dones, info = env.step(chosen_action)
            

            if(rendering and i % 1 == 0):
                env.render()

            # Added the new state to your list.
            explore_count.append(1)
            observations.append(obs)
            embedded_obs = embedding_function(obs, embedding_model)
            embeddings.append(embedded_obs)
            pickles.append(env.get_env_pickle())

            np.maximum(max_embedding_val, embedded_obs, out=max_embedding_val)
            np.minimum(min_embedding_val, embedded_obs, out=min_embedding_val)

            # Test it fit a gaussian fitted sampling.
            # Test it with mixture gaussians

        farthest_to_start = np.argmax([distance_measure(embeddings[0], em_obs) for em_obs in embeddings])
        print("Farthest to start index is:", farthest_to_start)
        chosen_pickle = pickles[farthest_to_start]
        env.load_env_pickle(chosen_pickle)
        farthest_points.append(chosen_pickle)

        observations = [observations[farthest_to_start]]
        embeddings = [embeddings[farthest_to_start]]
        pickles = [chosen_pickle]
        explore_count = [0]

    run_count = len(os.listdir('Farthest_Points'))
    np.save(f"Farthest_Points/smw_{run_count}.npy", farthest_points)
    
    if True:
        for state in farthest_points:
            env.load_env_pickle(state)
            chosen_action = [env.action_space.sample()]
            obs, rewards, dones, info = env.step(chosen_action)
            env.render()
            input("Press enter to go to the next")

    env.unwrapped.close()
    return embeddings, pickles

# Embedding Functions
def pix2mem(input: "numpy_array", model=None) -> "numpy_array":
    return model.predict(input).squeeze()

def no_embedding(input: "numpy_array", model=None) -> "numpy_array":
    return tuple(input)

# Author: Harmen Kang
def GO_explore_downsampling(input: "numpy_array", model=None) -> "numpy_array":
    print("input.shape:",input.shape)
    print("input.squeeze:",input.squeeze())
    isqueezed = input.squeeze()
    downsampled_image = block_reduce(isqueezed, block_size=(3,3,1), func=np.mean)
    return downsampled_image


# Action selection functions
def random_action():
    def random_action_setup(env: "OpenAI-Env"=None):
        info = dict()
        info["action_space"] = env.action_space
        return info
    
    def random_action_act(obs: "numpy_array", info: dict):
        return info["action_space"].sample()

    return random_action_setup, random_action_act

def learned_policy():
    def learned_policy_setup(env: "OpenAI-Env"=None):
        info = dict()
        info["model"] = PPO2.load("Model/Policy/SMW_level_one.pkl")
        return info
# 1986.5_SuperMarioWorld-Snes_model.pkl
    def learned_policy_act(obs: "numpy_array", info: dict):
        return info["model"].predict(obs)[0]

    return learned_policy_setup, learned_policy_act    

def chaos_monkey():
    def chaos_monkey_setup(env: "OpenAI-Env"=None):
        info = dict()
        info["ACTION_NUM"] = 4096

        info["distribution"] = np.load("Distributions/SNES_retro_dist.npy")

        info["used_actions"] = [[]]
        info["chosen_index"] = None
        return info

    def chaos_monkey_act(obs: "numpy_array", info: dict):
        # Select an action weighted by the loaded distribution.
        temp_action = np.random.choice(info["ACTION_NUM"], p=info["distribution"])

        # print(len(info["used_actions"]))
        # Convert the action int to a binary representation of that number
        str_action = f'{temp_action:b}'.rjust(12, '0')[::-1]
        
        # Make the list int
        action = [int(str_button) for str_button in str_action]
        return action

    return chaos_monkey_setup, chaos_monkey_act

def chaos_monkey_action_reduction():
    def chaos_monkey_setup(env: "OpenAI-Env"=None):
        info = dict()
        info["ACTION_NUM"] = 4096

        info["distribution"] = np.load("Distributions/SNES_retro_dist.npy")
        
        # Add a constant so every button has a chance to be picked
        info["distribution"] += 1 / 4096

        # Normalize so things sum up to 1
        info["distribution"] /= sum(info["distribution"])

        info["used_actions"] = [[]]
        info["chosen_index"] = None
        return info

    def chaos_monkey_act(obs: "numpy_array", info: dict):
        # Select an action weighted by the loaded distribution.
        temp_action = np.random.choice(info["ACTION_NUM"], p=info["distribution"])
        info["distribution"][temp_action] = 0
        info["distribution"] /= sum(info["distribution"])

        # print(len(info["used_actions"]))
        # Convert the action int to a binary representation of that number
        str_action = f'{temp_action:b}'.rjust(12, '0')[::-1]
        
        # Make the list int
        action = [int(str_button) for str_button in str_action]
        return action

    return chaos_monkey_setup, chaos_monkey_act

# Distance Measures
def euclidian_distance(pointA: "numpy_array", pointB: "numpy_array", _norm=np.linalg.norm) -> float:
    return _norm(pointA - pointB)


# Get Goal Functions
def get_random_goal_pix2mem(obs, max_val, min_val):
    rand = np.random.uniform(low=min_val, high=max_val).squeeze()
    return rand

def get_random_goal(obs, max_val, min_val):
    return np.random.uniform(size=obs.shape, low=min_val, high=max_val)


# Get Score
def get_score(embeddings):
    embeddings = np.array(embeddings)
    length = len(embeddings)
    bbox_sum = np.zeros(length-1)
    nuc_norm = np.zeros(length-1)
    for i in range(2, length):
        bbox_sum[i-1] = sum(np.amax(embeddings[:i], axis=0) - np.amin(embeddings[:i], axis=0))
        nuc_norm[i-1] = np.linalg.norm(np.cov(embeddings[:i].T), ord='nuc')

    return bbox_sum, nuc_norm
