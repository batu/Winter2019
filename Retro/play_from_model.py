import retro, gym
from keras.models import load_model

from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import Pix2Mem
from stable_baselines.common.cmd_util import make_retro_env
from stable_baselines.common.vec_env import VecFrameStack

model_name = "best_model_pix"
Pix2MemEmbedding = "pix2men" in model_name
atari = "atari" in model_name

if True:
    env = retro.make('SuperMarioWorld-Snes')
    env = Pix2Mem(env, "Embeddings/mario_1.h5")
elif atari:
    env = gym.make('Breakout-ram-v0')
else:
    env = make_retro_env('SuperMarioWorld-Snes', num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=1)

model = PPO2.load(f"SelectedModels/{model_name}.pkl")

# Enjoy trained agent
obs = env.reset()
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(info)
    print(rewards)
    if dones:
        env.reset()
    env.render()
