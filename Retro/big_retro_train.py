import retro, gym
import numpy as np
import os
import tensorflow as tf
import time, random, sys
import cloudpickle
import argparse
from support_utils import save_hyperparameters, parseArguments, send_email, visualize_run, visualize_experiment
import imageio, glob
import numpy as np
import six
import matplotlib.pyplot as plt

from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.cmd_util import make_retro_env, make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines.common.atari_wrappers import Pix2Mem, ScaledFloatFrame
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv, SnapshotVecEnv, DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines import DQN
from stable_baselines import A2C

def linear_schedule(initial_value):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    def func(progress):
        return progress * initial_value
    return func

# BREADCRUMBS_START
for run_num in range(20):
    for isExternal in [True]:
        NUM_CPU = 1
        seed = run_num
        ENV_NAME = "SuperMarioWorld-Snes"
        ENV_NAME = "BeamRiderNoFrameskip-v4"
        experiment_name = "IncreasingExternalBeamrider" if isExternal else "StandartBeamrider"
        # state_loading_dict = {"load_path":"States/SMW/smw_12_300.pickle",
        # state_loading_dict = {"load_path":"States/Breakout/breakout_250.pickle",
        #                       "policy":"random"}
        training_length = 50000000 
        test_count = 10

        # BREADCRUMBS_END
        set_global_seeds(seed)
        REPLAY = False
        TB_path = f"Results/Tensorboard/{experiment_name}/"

        run_number = 0
        try:
            os.mkdir(TB_path[:-1])
        except:
            pass

        try:
            os.mkdir(f"{TB_path}README")
        except:
            pass

        models_path = "Results/SavedModels/"

        changes = """The snapshots are manually recorded. The policy is random."""
        reasoning = """The introduction of 100 states might have introduced too much variance."""
        hypothesis = """Less variance and better training. """


        best_mean_reward, n_steps = -np.inf, 0

        last_name_ext = "dumm"
        last_name = "dummy"
        episode_counter = 0
        external_best = -np.inf
        def callback(_locals, _globals):
            """
            Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
            :param _locals: (dict)
            :param _globals: (dict)
            """
            global n_steps, best_mean_reward, last_name, episode_counter, external_best, last_name_ext

            # # Print stats every 1000 calls
            if (n_steps + 1) % 100 == 0:
                # Evaluate policy performance
                x, y = ts2xy(load_results(run_path), 'timesteps')
                if len(x) > 0:
                    mean_reward = np.mean(y[-100:])
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        best_mean_reward, mean_reward))

                    # New best model, you could save the agent here
                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        # Example for saving best model
                        if os.path.exists(run_path + f'/{last_name}'):
                            os.remove(run_path + f'/{last_name}')
                        print("Saving new best model")
                        last_name = f"{best_mean_reward:.1f}_{ENV_NAME}_model.pkl"
                        _locals['self'].save(
                            run_path + f'/{best_mean_reward:.1f}_{ENV_NAME}_model.pkl')
            n_steps += 1

            if any(_locals["masks"]):
                episode_counter += 1

            if ((episode_counter + 1) % test_count) == 0:
                episode_counter = 0
                obs = _locals["self"].env.reset()
                cumulative_rewards = 0
                while True:
                    action, _states = model.predict(obs)
                    obs, rewards, dones, _ = _locals["self"].env.step(action)
                    cumulative_rewards += rewards[0]
                    if dones:
                        with open(os.path.join(f"{run_path}", "test_rewards.txt"), "a+") as file:
                            file.write(f"{cumulative_rewards}\n")
                        break

                if cumulative_rewards > external_best:
                    external_best = cumulative_rewards
                    if os.path.exists(run_path + f'/{last_name_ext}'):
                        os.remove(run_path + f'/{last_name_ext}')
                    last_name_ext = f"{external_best:.1f}_{ENV_NAME}_external_model.pkl"
                    _locals['self'].save(
                        run_path + f'/{external_best:.1f}_{ENV_NAME}_external_model.pkl')
            return True


        folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
        run_name = f"run{folder_count}"
        run_path = f'{TB_path}{run_name}'
        os.mkdir(run_path)

        with open(os.path.join(f"{run_path}", "README.txt"), "w") as readme:
            start_time_ascii = time.asctime(time.localtime(time.time()))
            algorithm = os.path.basename(__file__)[:-2]
            print(f"Experiment start time: {start_time_ascii}", file=readme)
            print(f"\nAlgorithm:\n{algorithm}", file=readme)
            print(f"\nSeed:\n{seed}", file=readme)
            print(f"\nThe Changes:\n{changes}", file=readme)
            print(f"\nReasoning:\n{reasoning}", file=readme)
            print(f"\nHypothesis:\n{hypothesis}", file=readme)
            print(f"\nResults:\n", file=readme)

        # This function saves all the important hypterparameters to the run summary file.
        save_hyperparameters(["retro_train.py"], f"{run_path}/run_summary.txt", experiment_name=experiment_name)

        start_time_ascii = time.asctime(time.localtime(time.time()))
        start_time = time.time()
        print("Training has started!")
        # # multiprocess environment


        # env = Monitor(env, filename=run_path, allow_early_resets=True)
        # env = VecFrameStack(env, n_stack=4)
        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                net_arch=[dict(act_fun=tf.nn.leaky_relu, net_arch=[32, 32, 32])],
                feature_extraction="mlp")
                policy_kwargs = dict(act_fun=tf.nn.leaky_relu, net_arch=[32, 32, 32])
        # env = retro.make(f'{ENV_NAME}')
        # BREADCRUMBS_START
        # env = retro.make(f'{ENV_NAME}')#, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE)
        # env = Pix2Mem(env, "Embeddings/mario_1.h5")

        # env = retro.make(f'{ENV_NAME}', obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE)
        # env = Monitor(env, filename=run_path, allow_early_resets=True)
        if isExternal:
            env = gym.make(f"{ENV_NAME}")
            env = ScaledFloatFrame(env)
            env = Pix2Mem(env, "Embeddings/beamrider_encoder.h5")
            env = Monitor(env, filename=run_path, allow_early_resets=True)
            env = SnapshotVecEnv([lambda: env])
            env = VecNormalize(env, 4)
            # env = make_retro_env('SuperMarioWorld-Snes', num_env=1, seed=seed, logdir=run_path)
            model = PPO2(MlpPolicy, env, verbose=1, #tensorboard_log=f"{run_path}",
                        gamma=0.99,
                        n_steps=128,
                        ent_coef=0.01,
                        learning_rate=linear_schedule(0.00025),
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        lam=0.95,
                        nminibatches=4,
                        noptepochs=4,
                        cliprange=linear_schedule(0.2))
        else:
            env = make_atari_env(f"{ENV_NAME}", seed=seed, num_env=1)
            env = VecFrameStack(env, 4)
            model = PPO2(CnnPolicy, env, verbose=1, #tensorboard_log=f"{run_path}",
            gamma=0.99,
            n_steps=128,
            ent_coef=0.01,
            learning_rate=linear_schedule(0.00025),
            vf_coef=0.5,
            max_grad_norm=0.5,
            lam=0.95,
            nminibatches=4,
            noptepochs=4,
            cliprange=linear_schedule(0.2))

        model.learn(total_timesteps=training_length, callback=callback)
        # BREADCRUMBS_END
        print("The training has completed!")
        finish_time = time.time()
        model.save(f"{run_path}/{ENV_NAME}_final.pkl")
        visualize_run(run_path)
        del model
        del env
        send_email(f"The {experiment_name}, run {run_number} ended.", run_path)

visualize_experiment(TB_path)
send_email("Big log has ended!", TB_path)
