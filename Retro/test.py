import retro
env = retro.make('SuperMarioWorld-Snes')
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
