from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from src.nn_ops import *
import numpy as np


# TODO: 1. Add summary to max Q value, average Q value, and total reward for one episode.
# TODO: 2. Modify reply memory so it is able to store and sample high dimensional data.
# TODO: 3. Add batch normalization to actor and critic networks
# TODO: 4. Add high dimensional network build.
# TODO: 5. Create functions to save the model parameters (in DDPG) and replay memory (in Memory class).
# TODO: 6. Create testing function (in DDPG)
# TODO: 7. Create Policy class so that different policies can be implemented.

class DDPG(object):
    def __init__(self, sess, critic, actor, env, memory, gamma=.99, render=True):
        """
        Args:
            sess: tensorflow session variable
            critic: critic network, the output of the second dim should be num_actions
            actor: actor network, the output of the second dim should be 1
            memory: Replay Memory
        """

        # model variables
        self.sess = sess
        self.gamma = gamma

        # memory
        self.memory = memory
        self.batch_size = self.memory.batch_size

        # openai gym environment
        self.env = env

        # action bound
        self.action_bound = env.action_space.high

        # critic network
        self.critic = critic

        # actor network
        self.actor = actor

        self.render = render

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def fit(self, max_epoch, max_step=None):
        for epoch in range(0, max_epoch):
            current_state = self.env.reset()
            per_game_reward = 0
            step = 0
            done = False
            while not done:
                if self.render:
                    self.env.render()
                current_state = current_state.reshape(1, 3)
                action = self.actor_action(current_state) + (1. / (1. + epoch + step))
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, 3)
                per_game_reward += reward
                terminal = 0 if done else 1
                # store experiences
                self.memory.add(current_state, next_state, reward=reward, action=action, terminal=terminal)

                # train models
                if self.batch_size < self.memory.count:
                    # random sample
                    s, next_s, r, t, a = self.memory.sample()
                    a = a.reshape(self.batch_size, -1)

                    # clip the reward? Is this good?
                    r = np.tanh(r)

                    # step 1: target action
                    target_action = self.target_actor_action(next_s)

                    # step 2: estimate next state's Q value according to this action
                    y = self.critic_target_q(next_s, target_action)
                    y = r + t * self.gamma * y

                    # step 3: update critic
                    loss, _ = self.sess.run([self.critic.loss, self.critic.train],
                                            feed_dict={self.critic.y: y, self.critic.action: a, self.critic.x: s})
                    # step 4: perceive action according to actor given s
                    actor_action = self.actor_action(s)

                    # step 5: calculate action gradient
                    action_gradient = self.sess.run(self.critic.action_gradient,
                                                    {self.critic.x: s, self.critic.action: actor_action})
                    # step 6: update actor policy
                    self.sess.run(self.actor.train, {self.actor.x: s, self.actor.action_gradient: action_gradient[0]})

                    # update targets
                    self.sess.run([self.actor.update_op, self.critic.update_op])
                current_state = next_state

                # increase the step
                step += 1

                # if max step is provided, terminate at the max_step during every episode.
                if step == max_step:
                    done = True

            print("At game %s, total reward is %s" % (epoch, per_game_reward))

    def critic_network_q(self, state, action):
        return self.sess.run(self.critic.network, feed_dict={self.critic.action: action, self.critic.x: state})

    def critic_target_q(self, state, action):
        return self.sess.run(self.critic.target, feed_dict={self.critic.target_action: action,
                                                            self.critic.target_x: state})

    def actor_action(self, state):
        """
        Choose action based on current policy and current state
        """
        return self.sess.run(self.actor.network, feed_dict={self.actor.x: state}) * self.action_bound

    def target_actor_action(self, state):
        return self.sess.run(self.actor.target, feed_dict={self.actor.target_x: state}) * self.action_bound
