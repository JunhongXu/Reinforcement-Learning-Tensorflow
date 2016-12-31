from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.agent.core import BaseAgent
import numpy as np
from src.utilities import *
import os
from src.policies import *
from gym.wrappers import Monitor
import tensorflow as tf

# TODO: 1. Add summary to max Q value, average Q value, and total reward for one episode.
# TODO: 2. Modify reply memory so it is able to store and sample high dimensional data.
# TODO: 3. Add batch normalization to actor and critic networks
# TODO: 4. Add high dimensional network build.
# TODO: 7. Create Policy class so that different policies can be implemented.


class DDPG(BaseAgent):
    def __init__(self, sess, critic, actor, env, env_name, memory, max_step, normalizer=None, warm_up=5000, max_test_epoch=3, gamma=.99,
                 render=True):
        """
        A deep deterministic policy gradient agent.
        """
        super(DDPG, self).__init__(sess, env=env, memory=memory, gamma=gamma, render=render, normalizer=normalizer,
                                   max_step=max_step, env_name=env_name, warm_up=warm_up, max_test_epoch=max_test_epoch)

        self.policy_noise = OUNoise(actor.action_dim)

        # action bound
        self.action_bound = env.action_space.high

        # critic network
        self.critic = critic

        # actor network
        self.actor = actor

        self.env_high = self.env.observation_space.high
        self.env_low = self.env.observation_space.low

        directory = os.path.join("tmp", type(self).__name__)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # do not know if this is the right thing to do.. one for evaluation, one for training
        self.train_monitor = Monitor(self.env, os.path.join(directory, self.env_name), force=True, mode="training")
        self.test_monitor = Monitor(self.env, os.path.join(directory, self.env_name), force=True, mode="evaluation")

        self.restore()

        self.global_step = self.sess.run(self.step)
        self.is_training = True if self.global_step < self.max_step else False

    def fit(self, max_step_per_game=None):
        # evaluate step variable
        print("Starting from step %s" % self.global_step)

        global_epoch = 0
        # iterate until reach the maximum step
        while self.global_step < self.max_step:

            # re-initialize per game variables
            current_state = self.train_monitor.reset()

            # apply normalizer
            if self.normalizer:
                current_state = self.normalizer.normalize(current_state)

            per_game_reward = 0
            per_game_step = 0
            done = False
            self.policy_noise.reset()
            print("Progress: %s %s" % (progress(self.global_step, self.max_step, 50)[0],
                                       progress(self.global_step, self.max_step, 50)[1]))
            while not done:
                if self.render:
                    self.train_monitor.render()

                current_state = current_state.reshape(1, -1)

                # take noisy action
                action = self.action(current_state) + (self.policy_noise.noise() * self.action_bound)
                next_state, reward, done, _ = self.train_monitor.step(action)

                # apply normalizer
                if self.normalizer:
                    next_state = self.normalizer.normalize(next_state)

                next_state = next_state.reshape(1, -1)
                per_game_reward += reward

                terminal = 0 if done else 1

                # store experiences
                self.memory.add(current_state, next_state, reward=reward, action=action, terminal=terminal)

                # train models
                if self.global_step > self.warm_up:
                    if self.batch_size < self.memory.count:
                        # random sample
                        s, next_s, r, t, a = self.memory.sample()
                        a = a.reshape(self.batch_size, -1)

                        # step 1: target action
                        target_action = self.actor_target_predict(next_s)

                        # step 2: estimate next state's Q value according to this action (target)
                        y = self.critic_target_predict(next_s, target_action)
                        y = r + t * self.gamma * y

                        # step 3: update critic
                        loss, _ = self.sess.run([self.critic.loss, self.critic.train],
                                                feed_dict={self.critic.y: y, self.critic.action: a, self.critic.x: s})
                        # step 4: perceive action according to actor given s
                        actor_action = self.action(s)

                        # step 5: calculate action gradient
                        action_gradient = self.sess.run(self.critic.action_gradient,
                                                        {self.critic.x: s, self.critic.action: actor_action})
                        # step 6: update actor policy
                        self.sess.run(self.actor.train, {self.actor.x: s,
                                                         self.actor.action_gradient: action_gradient[0]})

                        # update targets
                        self.sess.run([self.actor.update_op, self.critic.update_op])
                        # end of training

                if self.global_step % 5000 == 0 and self.global_step != 0:
                    self.save()

                current_state = next_state

                # if max step is provided, terminate at the max_step during every episode.
                if max_step_per_game and per_game_step % max_step_per_game == 0 and per_game_step != 0:
                    done = True

                # increase global step and step variable
                per_game_step += 1
                self.global_step += 1
                self.sess.run(self.step_assign)

            if global_epoch % 10 == 0 and global_epoch != 0:
                self.evaluate()
            global_epoch += 1
            print("At step %s, total reward is %s" % (self.global_step, per_game_reward))

        self.train_monitor.close()
        # if we reach the max step, evaluate the model
        self.evaluate()

    def evaluate(self):

        print("Start evaluating")
        # only test for 1 time if not reach the max training epoch
        max_test_epoch = self.max_test_epoch if not self.is_training else 1
        total_reward = 0
        for epoch in range(max_test_epoch):
            # start the environment
            state = self.test_monitor.reset()
            step = 0
            done = False
            # start one epoch
            while not done:
                self.test_monitor.render()

                if self.normalizer:
                    state = self.normalizer.normalize(state)

                action = self.action(state.reshape(1, -1))
                state, reward, done, _ = self.test_monitor.step(action)
                total_reward += reward
                step += 1
        print("Evaluation reward is %s" % (total_reward/max_test_epoch))
        self.test_monitor.close()

    def critic_predict(self, state, action):
        return self.sess.run(self.critic.network,
                             feed_dict={self.critic.action: action, self.critic.x: state})

    def critic_target_predict(self, state, action):
        return self.sess.run(self.critic.target,
                             feed_dict={self.critic.target_action: action, self.critic.target_x: state})

    def action(self, state):
        """
        For evaluation without noise
        """
        return (self.sess.run(self.actor.network,
                              feed_dict={self.actor.x: state})) * self.action_bound

    def actor_target_predict(self, state):
        return self.sess.run(self.actor.target, feed_dict={self.actor.target_x: state}) * self.action_bound
