import retro
from stable_baselines.common.vec_env import SnapshotVecEnv
import numpy as np
import pickle
# movie = retro.Movie('Movies/breakout_savestate_start.bk2')
# movie = retro.Movie('Movies/SMW_level_1_2.bk2')new_clean_smw1
movie = retro.Movie('Movies/gods_hand.bk2')
movie.step()

STATE_REPLAY = False

game_name = "SuperMarioWorld-Snes"
# game_name = "Breakout-Atari2600"
env = retro.make(game=game_name, state = None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
# env = retro.make(game="Breakout-Atari2600", state="/home/batu/Desktop/Winter_2019/Retro/States/Breakout/breakout_start_game.state", use_restricted_actions=retro.Actions.ALL, players=movie.players)
env.initial_state = movie.get_state()
env.reset()
num_buttons = env.num_buttons
# env = SnapshotVecEnv([lambda:env])




print("1 = 1, 2 = 10, 3 = 100, 4 = 1000, frames forward.")
state_save_name = "SMW_l12_starts"
if not STATE_REPLAY:

    states = []
    total_frames = 0
    state_save_name = input("What should the name of the states be? ")
    while True:
        frames_forward = 0
        inp = input('Enter your input: ')
        try:
            number_input = float(inp)
            frames_forward = int(10 ** (number_input - 1))
            total_frames += frames_forward
            print(f"Forwarding {frames_forward} frames, going to frame: {total_frames}.")
        except:
            pass
        if inp == "q":
            break
        if inp == "s":
            states.append(env.em.get_state())
            # states.append(env.get_env_pickle())
            print("State appended!")
        if inp == "r":
            save_counter = int(input("Once how many frames should I save?"))
            counter = save_counter
            while movie.step():
                keys = []
                if counter == save_counter:
                    states.append(env.em.get_state())
                    # state = env.get_env_pickle()
                    print(f"Saved the {len(states)}th state at frame {total_frames}.")
                    counter = 0
                for p in range(movie.players):
                    for i in range(env.num_buttons):
                        keys.append(movie.get_key(i, p))
                _obs, _rew, _done, _info = env.step(keys)
                total_frames += 1
                env.render()
                counter += 1
            else:
                print("Done saving.")
                break
        for _ in range(frames_forward):
            movie.step()
            keys = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    keys.append(movie.get_key(i, p))
            _obs, _rew, _done, _info = env.step(keys)
            env.render()

    print("The state saving has been completed.")
    print(f"There has been {len(states)} in total.")
    # states = np.array(states)
    with open(f'States/{state_save_name}.pickle', 'wb') as f:
        pickle.dump(states, f,  pickle.HIGHEST_PROTOCOL)

with open(f'States/{state_save_name}.pickle', 'rb') as f:
    new_states = pickle.load(f)
print(f"Saved at {state_save_name}!")
del env

env = retro.make(game=game_name)
env = SnapshotVecEnv([lambda:env], training=False)
env.reset()
for state in range(10000):
    env.render()
    input("Press enter to go to the next")
    env.load_env_pickle(new_states[state])
    chosen_action = [env.action_space.sample()]
    obs, rewards, dones, info = env.step(chosen_action)


env.close()
