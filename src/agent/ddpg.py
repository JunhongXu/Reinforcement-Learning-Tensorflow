from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from src.nn_ops import *


class DDPG(object):
    def __init__(self, sess, critic, actor, env, memory):
        """
        Args:
            sess: tensorflow session variable
            critic: critic network, the output of the second dim should be num_actions
            actor: actor network, the output of the second dim should be 1
            memory: Replay Memory
        """

        # model variables
        self.sess = sess

        # openai gym environment
        self.env = env

        # critic network
        self.critic = critic

        # actor network
        self.actor = actor
        self.memory = memory

        # initialize variables
        self.sess.run(tf.global_variables_initializer())


