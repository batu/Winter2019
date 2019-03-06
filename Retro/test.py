import retro
import gym 
import numpy as np
class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        # ['BUTTON', None, 'SELECT', 'RESET', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()




env = retro.make('Breakout-Atari2600')
env = Discretizer(env, combos=[["BUTTON"], ["LEFT"], ["RIGHT"]])
obs = env.reset()
print(env.unwrapped.buttons)
# ['BUTTON', None, 'SELECT', 'RESET', 'UP', 'DOWN', 'LEFT', 'RIGHT']
for _ in range(5000):
    stuff = env.step(env.action_space.sample())
    env.render()
    # print(stuff)
