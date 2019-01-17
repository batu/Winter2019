import random
import numpy as np


def run_rrt(env, embedding_function, action_policy, distance_measure, get_goal, action_steps=8, max_steps = 10000, rendering=True, state_pruning=True):

    unwrapped_env = env.envs[0].unwrapped
    observation_space = env.reset().shape
    action_space = unwrapped_env.action_space

    # assert type(env) == SnapshotVecEnv
    obs = env.reset()

    observations = [obs]
    embeddings = [embedding_function(obs)]
    pickles = [env.get_env_pickle()]

    for i in range(max_steps):
        goal = get_goal(obs)
        chosen_index = np.argmin([distance_measure(em_obs, goal) for em_obs in embeddings])
        #print(chosen_index)

        chosen_pickle = pickles[chosen_index]
        env.load_env_pickle(chosen_pickle)

        chosen_observation = observations[chosen_index]
        chosen_action = [action_policy(chosen_observation, action_space)]

        for _ in range(action_steps):
            obs, rewards, dones, info = env.step(chosen_action)
            if(any(dones)):
                obs = env.reset()
                print("Env reset!");

        if(rendering and i % 50 == 0):
            env.render()

        if state_pruning:
            test_arr = np.asarray(observations)
            prune = np.all(np.isin(obs[0], test_arr))
            if not prune:
                observations.append(obs)
                embeddings.append(embedding_function(obs))
                pickles.append(env.get_env_pickle())
        else:
            observations.append(obs)
            embeddings.append(embedding_function(obs))
            pickles.append(env.get_env_pickle())
    env.unwrapped.close()

# Embedding Functions
def no_embedding(input: "numpy_array") -> "numpy_array":
    return input

def GO_explore_downsampling(input: "numpy_array") -> "numpy_array":
    return


# Action selection functions
def random_action(obs: "numpy_array", action_space: int) -> int:
    return random.randint(0, action_space.n -1)


# Distance Measures
def euclidian_distance(pointA: "numpy_array", pointB: "numpy_array", _norm=np.linalg.norm) -> float:
    return _norm(pointA - pointB)


# Get Goal Functions
def get_random_goal(obs):
    return np.random.rand(*obs.shape) * 2 - 1
