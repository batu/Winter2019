import retro

movie = retro.Movie('Movies/breakout_savestate_start.bk2')
movie.step()

env = retro.make(game="Breakout-Atari2600", state="/home/batu/Desktop/Winter_2019/Retro/States/Breakout/breakout_start_game.state", use_restricted_actions=retro.Actions.ALL, players=movie.players)
# env.initial_state = movie.get_state()
env.reset()

while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    _obs, _rew, _done, _info = env.step(keys)
    env.render()
