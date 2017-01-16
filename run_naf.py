import tensorflow as tf
from src.networks import NAFNetwork
from src.policies import OUNoise
from src.env_wrapper import NormalizeWrapper
from src.replay import Memory
from src.agent.naf import NAF
import gym


# LEARNING RATE
LEARNING_RATE = 1e-3
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
ENV_NAME = "BipedalWalker-v2"

env = gym.make(ENV_NAME)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape

network = NAFNetwork(action_dim=action_dim, input_dim=state_dim, name="NAF", tau=TAU,
                     optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE), use_bn=False)
memory = Memory(1000000, state_dim, action_dim, 64)
policy = OUNoise(action_dim)
# env = NormalizeWrapper(env, -1, 1)
with tf.Session() as sess:
    agent = NAF(sess, network, env, ENV_NAME, max_step=10000000, evaluate_every=100, save_every=15000, memory=memory, policy=policy,
                record=True, render=False, warm_up=10000)
    agent.fit()
