import tensorflow as tf
from src.agent.ddpg import *
from src.networks import *
from src.replay import *
import gym

# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 2000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
ENV_NAME = "MountainCarContinuous-v0"


env = gym.make("MountainCarContinuous-v0")
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
critic = CriticNetwork(action_dim=action_dim, input_dim=[state_dim],
                       optimizer=tf.train.AdamOptimizer(CRITIC_LEARNING_RATE), tau=TAU)
actor = ActorNetwork(action_dim=action_dim, input_dim=[state_dim],
                     optimizer=tf.train.AdamOptimizer(ACTOR_LEARNING_RATE), tau=TAU)

memory = Memory(10000, "", state_dim, action_dim, 64)

with tf.Session() as sess:
    agent = DDPG(sess, critic, actor, render=True, env=env, memory=memory, max_step=10000, env_name=ENV_NAME)
    agent.fit(MAX_EP_STEPS)