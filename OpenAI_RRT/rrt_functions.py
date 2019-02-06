import random
import numpy as np

from keras.models import load_model
from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import Pix2Mem
from stable_baselines.common.cmd_util import make_retro_env
from stable_baselines.common.vec_env import VecFrameStack

# This is the main function that starts the rrt search.
def run_rrt(env, embedding_function, action_policy, distance_measure, get_goal,
action_steps=30, max_steps = 750, verbose=True, rendering=True, state_pruning=False, visitation_weight=10.0):

    # Extracts information about the environment
    unwrapped_env = env.envs[0].unwrapped

    # This can pixel input. This can be RAM, this can be anything
    observation_space = env.reset().shape

    # [1.1.0.0.0.1.0.1.0.1]
    # Action space looks like this. Digits map to a button on a controller. 1 means pressed 0 means not pressed.
    action_space = unwrapped_env.action_space

    embedding_model = None
    action_policy_model = None

    # This loads the embedding if necesary
    if embedding_function.__name__ == "pix2mem":
        embedding_model = load_model("Model/smw_pix2mem.h5")

    # This loads the model if necesary
    if action_policy.__name__ == "learned_policy":
        action_policy_model = PPO2.load("Model/Policy/best_model_pix.pkl")

    # assert type(env) == SnapshotVecEnv
    obs = env.reset()

    # Keep track of the states/embeddings/pickles
    observations = [obs]
    embeddings = [embedding_function(obs, embedding_model)]
    pickles = [env.get_env_pickle()]
    explore_count = [0]

    for i in range(max_steps):
        # Get a goal
        goal = get_goal(embedding_function(obs, embedding_model))

        # Find which state is closest to your goal
        chosen_index = np.argmin([distance_measure(em_obs, goal) *
                                 (1 + (explore_count[index] + 1) / visitation_weight)
                                  for index, em_obs in enumerate(embeddings)])

        # Count how many times you chose that state. This helps reduce the duplicate choices.
        explore_count[chosen_index] += 1
        if verbose:
            print(f"In step {i} the chosen index is: {chosen_index}")

        # Load the closest goal.
        chosen_pickle = pickles[chosen_index]
        env.load_env_pickle(chosen_pickle)

        # Get the correct embedding
        chosen_observation = embedding_function(observations[chosen_index], embedding_model)

        # Get the action to do from that chosen goal state.
        chosen_action = [action_policy(chosen_observation, action_space, action_policy_model)]

        # Take that action action step times (usually 30)
        for _ in range(action_steps):
            obs, rewards, dones, info = env.step(chosen_action)
            if(any(dones)):
                obs = env.reset()
                print("Env reset!");

        if(rendering and i % 1 == 0):
            env.render()

        # This is buggy. But the idea is to not add the state to the table if it already is in it.
        if state_pruning:
            test_arr = np.asarray(observations)
            prune = (obs[0] == test_arr).all()
            if not prune:
                explore_count.append(0)
                observations.append(obs)
                embeddings.append(embedding_function(obs, embedding_model))
                pickles.append(env.get_env_pickle())
        else:
            # Added the new state to your list.
            explore_count.append(0)
            observations.append(obs)
            embeddings.append(embedding_function(obs, embedding_model))
            pickles.append(env.get_env_pickle())
    env.unwrapped.close()
    return embeddings, pickles

# Embedding Functions
def pix2mem(input: "numpy_array", model=None) -> "numpy_array":
    return model.predict(input).squeeze()

def no_embedding(input: "numpy_array", model=None) -> "numpy_array":
    return input

def GO_explore_downsampling(input: "numpy_array", model=None) -> "numpy_array":
    return


# Action selection functions
def random_action(obs: "numpy_array", action_space: int, model=None):
    return action_space.sample()

def learned_policy(obs: "numpy_array", action_space: int, model=None):
    return model.predict(obs)[0]


# Distance Measures
def euclidian_distance(pointA: "numpy_array", pointB: "numpy_array", _norm=np.linalg.norm) -> float:
    return _norm(pointA - pointB)


# Get Goal Functions
def get_random_goal_pix2mem(obs):
    # FIX THE BOX SIZE.
    return np.random.uniform(size=256, low=min(obs), high=max(obs))

def get_random_goal(obs):
    return np.random.uniform(size=obs.shape, low=min(obs), high=max(obs))


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
