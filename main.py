import tensorflow as tf
from src.agent.ddpg import *
from src.networks import *
from src.replay import *
import gym
from src.env_wrapper import *
from gym.wrappers import TimeLimit

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
ENV_NAME = "BipedalWalker-v2"


env = gym.make(ENV_NAME)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape
critic = CriticNetwork(action_dim=action_dim, input_dim= state_dim,
                       optimizer=tf.train.AdamOptimizer(CRITIC_LEARNING_RATE), tau=TAU)
actor = ActorNetwork(action_dim=action_dim, input_dim= state_dim,
                     optimizer=tf.train.AdamOptimizer(ACTOR_LEARNING_RATE), tau=TAU)
policy = OUNoise(action_dim)
memory = Memory(1000000, state_dim, action_dim, 64)
# env = NormalizeWrapper(env, -1, 1)

# env = TimeLimit(env)
with tf.Session() as sess:
    agent = DDPG(sess, critic, actor, env=env, evaluate_every=100, max_test_epoch=100, warm_up=20000, policy=policy,
                 render=False, record=True, memory=memory, max_step=10000000, env_name=ENV_NAME)
    agent.fit()
