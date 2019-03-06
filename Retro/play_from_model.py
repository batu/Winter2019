import retro, gym
from keras.models import load_model

from stable_baselines import PPO2
from stable_baselines.common.atari_wrappers import Pix2Mem, ScaledFloatFrame, PixAB2C
from stable_baselines.common.cmd_util import make_retro_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.vec_env import DummyVecEnv

model_name = "LevelRepeater"
model_name = "1.8_BreakoutNoFrameskip-v4_model"
# model_name = "104.0_Breakout-Atari2600_model"

Pix2MemEmbedding = "pix2men" in model_name or True
atari = "atari" in model_name

if False:
    env = retro.make('SuperMarioWorld-Snes')#, use_restricted_actions=retro.Actions.DISCRETE)
    env = Pix2Mem(env, "Embeddings/mario_1.h5")
elif True:
    env = gym.make('BreakoutNoFrameskip-v4')
    env = ScaledFloatFrame(env)
    env = PixAB2C(env, "Embeddings/breakout_pixab2cmem.h5")
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

else:
    env = make_retro_env('SuperMarioWorld-Snes', num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=1)

model = PPO2.load(f"SelectedModels/{model_name}.pkl")

# Enjoy trained agent
obs = env.reset()
for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step([action])
    print(action)
    if dones:
        env.reset()
    env.render()
