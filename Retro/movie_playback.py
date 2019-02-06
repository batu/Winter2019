import retro

import numpy as np
movie = retro.Movie('Movies/play_session_1.bk2')
movie.step()

game_name = "SuperMarioWorld-Snes"
state_save_name = "every_100_from_level_1"
env = retro.make(game=game_name, state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
env.initial_state = movie.get_state()
env.reset()


print("1 = 1, 2 = 10, 3 = 100, 4 = 1000, frames forward.")

states = []
total_frames = 0
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
        print("State appended!")
    if inp == "r":
        save_counter = int(input("Once how many frames should I save?"))
        counter = save_counter
        while movie.step():
            keys = []
            if counter == save_counter:
                states.append(env.em.get_state())
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
np.save(f'States/{state_save_name}.npy', states)
print(f"Saved at {state_save_name}!")
env.close()
