import random
import numpy as np

from scipy.misc import imsave

from keras.models import load_model
from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import Pix2Mem
from stable_baselines.common.cmd_util import make_retro_env
from stable_baselines.common.vec_env import VecFrameStack

import sys

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

    max_embedding_val = embeddings[0]
    min_embedding_val = embeddings[0]

    hashes = {hash(embeddings[0])}

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
        # for _ in range(random.randint(4, action_steps)):
        for _ in range(int(action_steps / 2)):
            chosen_action = [action_policy_act(chosen_observation, info_dict)] 
            for _ in range(4):
                obs, rewards, dones, info = env.step(chosen_action)
                reset = any(dones)
                # if(any(dones)):
                #     # obs = env.reset()
                #     print("Env reset!")
        

        if(rendering and i % 1 == 0):
            env.render()

        # Added the new state to your list.
        if not reset:
            explore_count.append(1)
            observations.append(obs)
            embedded_obs = embedding_function(obs, embedding_model)
            embeddings.append(embedded_obs)
            pickles.append(env.get_env_pickle())

        max_embedding_val = np.maximum.reduce([max_embedding_val,embedded_obs])
        min_embedding_val = np.minimum.reduce([min_embedding_val,embedded_obs])

        # Test it fit a gaussian fitted sampling.
        # Test it with mixture gaussians

    print(explore_count)
    print(    max(explore_count))
    print(    min(explore_count))
    print(len(explore_count))
    env.unwrapped.close()
    return embeddings, pickles

# Embedding Functions
def pix2mem(input: "numpy_array", model=None) -> "numpy_array":
    return tuple(model.predict(input).squeeze())

def no_embedding(input: "numpy_array", model=None) -> "numpy_array":
    return tuple(input)

def GO_explore_downsampling(input: "numpy_array", model=None) -> "numpy_array":
    return


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

        info["REPLACEMENT"] = False
        info["distribution"] = np.load("Distributions/SNES_retro_dist.npy")
        
        if info["REPLACEMENT"]:
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
        if info["REPLACEMENT"]:
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
    return np.random.uniform(size=256, low=min_val, high=max_val)

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
