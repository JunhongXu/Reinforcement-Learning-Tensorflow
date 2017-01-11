from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.agent.core import BaseAgent
from src.utilities import *
import os
from src.policies import *
from gym.wrappers import Monitor
import tensorflow as tf

# TODO: 3. Add batch normalization to actor and critic networks
# TODO: 7. Create Policy class so that different policies can be implemented.


class DDPG(BaseAgent):
    def __init__(self, sess, critic, actor, env, env_name, memory, max_step, use_bn=False, record=True, warm_up=500,
                 max_test_epoch=3, gamma=.99, evaluate_every=100, render=True):
        """
        A deep deterministic policy gradient agent.
        """
        super(DDPG, self).__init__(sess, env=env, memory=memory, gamma=gamma, render=render, max_step=max_step,
                                   env_name=env_name, warm_up=warm_up, record=record, evaluate_every=evaluate_every,
                                   max_test_epoch=max_test_epoch)

        self.use_bn = use_bn

        self.policy_noise = OUNoise(actor.action_dim)

        # action bound
        self.action_bound = env.action_space.high

        # critic network
        self.critic = critic

        # actor network
        self.actor = actor

        # restore the model and replay memory if there is any
        self.restore()

        self.global_step = self.sess.run(self.step)
        self.global_epoch = self.sess.run(self.epoch)
        self.is_training = True if self.global_step < self.max_step else False
        self.q_summary = tf.summary.scalar("Q_Value", self.critic.network)

    def fit(self):
        print("Starting from step %s" % self.global_step)

        average_reward = 0

        # iterate until reach the maximum step
        while self.global_step < self.max_step:
            # evaluate model
            if self.global_epoch % self.evaluate_every == 0 and self.global_epoch != 0:
                self.evaluate()

            # re-initialize per game variables
            current_state = self.monitor.reset()
            current_state = current_state[np.newaxis]
            per_game_step = 0
            per_game_reward = 0
            done = False
            self.policy_noise.reset()
            print("Epoch %s" % self.global_epoch)
            print("Memory count %s" % self.memory.count)
            print("Progress: %s %s" % (progress(self.global_step, self.max_step, 100)[0],
                                       progress(self.global_step, self.max_step, 100)[1]))
            while not done:
                if self.render:
                    self.monitor.render()

                if self.warm_up > self.memory.count:
                    # take a random action to fill up the memory
                    action = self.monitor.action_space.sample()
                else:
                    # take a noisy action, exploration
                    action = self.action(current_state, False) + (self.policy_noise.noise() * self.action_bound)

                action = action.flatten()
                # evaluate the q value
                summary_q = self.critic_predict(current_state, action.reshape(1, -1), summary=True, is_train=False)
                self.writer.add_summary(summary_q, global_step=self.global_step)
                next_state, reward, done, _ = self.monitor.step(action)
                next_state = next_state[np.newaxis]
                average_reward += reward
                per_game_reward += reward

                terminal = 0 if done else 1

                # store experiences
                self.memory.add(current_state, next_state, reward=reward, action=action, terminal=terminal)

                # train models
                if self.warm_up < self.memory.count:
                    if self.batch_size < self.memory.count:
                        # random sample
                        s, next_s, r, t, a = self.memory.sample()
                        a = a.reshape(self.batch_size, -1)

                        # step 1: target action
                        target_action = self.actor_target_predict(next_s, is_train=True)

                        # step 2: estimate next state's Q value according to this action (target)
                        y = self.critic_target_predict(next_s, target_action, is_train=True)
                        y = r + t * self.gamma * y
                        if self.use_bn:
                            # step 3: update critic
                            loss, _ = self.sess.run([self.critic.loss, self.critic.train],
                                                    feed_dict={self.critic.network_train: True, self.critic.y: y,
                                                               self.critic.action: a, self.critic.x: s})

                            # step 4: perceive action according to actor given s
                            actor_action = self.action(s, True)

                            # step 5: calculate action gradient
                            action_gradient = self.sess.run(self.critic.action_gradient,
                                                            {self.critic.x: s, self.critic.action: actor_action,
                                                             self.critic.network_train: True})
                            # step 6: update actor policy
                            self.sess.run(self.actor.train, {self.actor.x: s,
                                                             self.actor.action_gradient: action_gradient[0],
                                                             self.actor.network_train: True})
                        else:
                                                        # step 3: update critic
                            loss, _ = self.sess.run([self.critic.loss, self.critic.train],
                                                    feed_dict={self.critic.y: y,
                                                               self.critic.action: a, self.critic.x: s})

                            # step 4: perceive action according to actor given s
                            actor_action = self.action(s, True)

                            # step 5: calculate action gradient
                            action_gradient = self.sess.run(self.critic.action_gradient,
                                                            {self.critic.x: s, self.critic.action: actor_action,})
                            # step 6: update actor policy
                            self.sess.run(self.actor.train, {self.actor.x: s,
                                                             self.actor.action_gradient: action_gradient[0],
                                                            })

                        # update targets
                        self.sess.run([self.actor.update_op, self.critic.update_op])
                        # end of training

                if self.global_step % 15000 == 0 and self.global_step != 0:
                    self.save()

                current_state = next_state
                per_game_step += 1
                # increase global step and step variable
                self.global_step += 1
                self.sess.run(self.step_assign)

            self.global_epoch += 1
            self.sess.run(self.epoch_assign)

            if self.global_epoch % self.evaluate_every == 0:
                print("At training epoch %s, average reward is %s" % (self.global_epoch,
                                                                      average_reward/self.evaluate_every))
                self.summary.value.add(tag="Average_reward", simple_value=float(average_reward/self.evaluate_every))
                self.writer.add_summary(self.summary, global_step=self.global_step)
                # refresh average reward
                average_reward = 0

        self.monitor.close()
        # if we reach the max step, evaluate the model
        self.is_training = False
        self.evaluate()

    def evaluate(self):
        monitor = self.monitor
        print("Evaluating")
        # only test for 1 time if not reach the max training epoch
        max_test_epoch = self.max_test_epoch if not self.is_training else 1
        total_reward = 0
        for epoch in range(max_test_epoch):
            # start the environment
            state = monitor.reset()
            step = 0
            done = False
            # start one epoch
            while not done:
                self.monitor.render()
                action = self.action(state[np.newaxis], False)
                state, reward, done, _ = monitor.step(action.flatten())
                total_reward += reward
                step += 1
        print("Average evaluation reward is %s" % (total_reward/max_test_epoch))
        if not self.is_training:
            self.monitor.close()

    def critic_predict(self, state, action, summary=False, is_train=None):
        """
        If summary is True, we also get the q value. This is used for logging.
        """
        if not self.use_bn:
            if summary:
                return self.sess.run(self.q_summary, feed_dict={self.critic.action: action, self.critic.x: state})
            else:
                return self.sess.run(self.critic.network, feed_dict={self.critic.action: action, self.critic.x: state})
        else:
            if summary:
                return self.sess.run(self.q_summary, feed_dict={self.critic.action: action, self.critic.x: state,
                                                                self.critic.network_train: is_train})
            else:
                return self.sess.run(self.critic.network, feed_dict={self.critic.action: action, self.critic.x: state,
                                                                self.critic.network_train: is_train})

    def critic_target_predict(self, state, action, is_train=None):
        if not self.use_bn:
            return self.sess.run(self.critic.target,
                                 feed_dict={self.critic.target_action: action, self.critic.target_x: state})
        else:
            return self.sess.run(self.critic.target, feed_dict={self.critic.target_action: action,
                                                                self.critic.target_x: state,
                                                                self.critic.target_train: is_train})

    def action(self, state, is_train=None):
        """
        For evaluation without noise
        """
        if not self.use_bn:
            return (self.sess.run(self.actor.network,
                                  feed_dict={self.actor.x: state})) * self.action_bound
        else:
            return (self.sess.run(self.actor.network,
                                  feed_dict={self.actor.x: state, self.actor.network_train: is_train})) * self.action_bound

    def actor_target_predict(self, state, is_train=None):
        if not self.use_bn:
            return self.sess.run(self.actor.target, feed_dict={self.actor.target_x: state}) * self.action_bound
        else:
            return self.sess.run(self.actor.target, feed_dict={self.actor.target_x: state,
                                                               self.actor.target_train: is_train}) * self.action_bound
