import retro

import numpy as np
movie = retro.Movie('Movies/play_session_1.bk2')
movie.step()

game_name = "SuperMarioWorld-Snes"
state_save_name = "smw_two_level_states"
env = retro.make(game=game_name, state=None, use_restricted_actions=retro.Actions.ALL, players=movie.players)
env.initial_state = movie.get_state()
env.reset()

manual = True


print("1 = 1, 2 = 10, 3 = 100, 4 = 1000, frames forward.")

states = []
frames_forward = 1
while True:
    if manual:
        try:
            button = float(input('Enter your input: '))
        except:
            continue
        if button == 9:
            break
        frames_forward = int(10 ** (button - 1))
        print(frames_forward)
    if frames_forward < 1:
        states.append(env.em.get_state())
        print("State appended!")
        frames_forward = 0
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
