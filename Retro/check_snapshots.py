import retro
import numpy as np
import pickle
from stable_baselines.common.vec_env import SnapshotVecEnv

game_name = "SuperMarioWorld-Snes"
state_save_name = "States/clean_smw1.pickle"

with open(f'{state_save_name}', 'rb') as f:
    states = pickle.load(f)

env = retro.make(game=game_name)
env = SnapshotVecEnv([lambda: env])
env.environment_category = "Retro"

env.reset()
for state in states:
    input("Press enter to go to the next")
    env.load_env_pickle(state)
    env.render()
    obs, _, _, _ = env.step([env.action_space.sample()])
print("Done.")
