import gym
import tensorflow as tf
import keras

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines import PPO2
import stable_baselines 

print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)
print("Stable Baselines version:", stable_baselines.__version__)

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.tanh

            extracted_features = tf.layers.flatten(self.processed_obs)

            print(extracted_features.shape)
            pi_h = extracted_features
            for i, layer_size in enumerate([64, 64]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([64, 64]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None        
        self._setup_init()
        tf.summary.FileWriter('logs/custom_policy_graph', tf.get_default_graph())


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})

class KerasPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(KerasPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        with tf.variable_scope("model", reuse=reuse):
            # self.processed_obs = self.obs_ph
            flat = tf.keras.layers.Flatten()(self.obs_ph)

            x = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc0')(flat)
            pi_latent = tf.keras.layers.Dense(64, activation="tanh", name='pi_fc1')(x)

            x1 = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc0')(flat)
            vf_latent = tf.keras.layers.Dense(64, activation="tanh", name='vf_fc1')(x1)

            value_fn = tf.keras.layers.Dense(1, name='vf')(vf_latent)

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()
        tf.summary.FileWriter('logs/keras_policy_graph', tf.get_default_graph())


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})


env = gym.make("CartPole-v0")
env = DummyVecEnv([lambda: env])

keras_model = PPO2(KerasPolicy, env, verbose=1)
keras_model.learn(total_timesteps=25000)
# Mean Episode Reward ~= 20


# model = PPO2(CustomPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# Mean Episode Reward > 100

