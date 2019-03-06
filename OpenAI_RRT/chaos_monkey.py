import numpy as np
import os
import retro
ENV_NAME = "SuperMarioWorld-Snes"

distribution = np.load("/home/batu/Desktop/Winter_2019/OpenAI_RRT/Distributions/SNES_distribution.npy")
ACTION_NUM = 4096
temp_action = np.random.choice(ACTION_NUM, p=distribution)
temp_action = np.argmax(distribution)
print(temp_action)
str_action = f'{temp_action:b}'.rjust(12, '0')[::-1]
action = [int(str_button) for str_button in str_action]
    
#["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"],
env = retro.make(ENV_NAME)

print(env.buttons)
obs = env.reset()
for _ in range(10000):
    temp_action = np.random.choice(ACTION_NUM, p=distribution)
    str_action = f'{temp_action:b}'.rjust(12, '0')[::-1]
    action = [int(str_button) for str_button in str_action]

    for _ in range(4):
        obs, rewards, dones, info = env.step(action)
        env.render()
        if(dones):
            obs = env.reset()
            print("Env reset!")