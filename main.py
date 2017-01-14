import tensorflow as tf
from src.agent.ddpg import *
from src.networks import *
from src.replay import *
import gym
from src.env_wrapper import *

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
ENV_NAME = "MountainCarContinuous-v0"


env = gym.make(ENV_NAME)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape
critic = CriticNetwork(action_dim=action_dim, input_dim= state_dim,
                       optimizer=tf.train.AdamOptimizer(CRITIC_LEARNING_RATE), tau=TAU)
actor = ActorNetwork(action_dim=action_dim, input_dim= state_dim,
                     optimizer=tf.train.AdamOptimizer(ACTOR_LEARNING_RATE), tau=TAU)
policy = OUNoise(action_dim)
memory = Memory(1000000, state_dim, action_dim, 64)
env = NormalizeWrapper(env, -1, 1)
with tf.Session() as sess:
    agent = DDPG(sess, critic, actor, env=env, max_test_epoch=200, warm_up=10000, policy=policy,
                 render=True, memory=memory, max_step=1000000, env_name=ENV_NAME)
    agent.fit()
