from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from src.agent.core import BaseAgent
import numpy as np


class NAF(BaseAgent):
    def __init__(self, sess, network, env, env_name, max_step, memory, policy, update_step=5, record=True, warm_up=5000,
                 max_test_epoch=100, gamma=.99, evaluate_every=100, render=True):

        super(NAF, self).__init__(sess, memory, env, env_name, max_step, policy, update_step=update_step,
                                  record=record, warm_up=warm_up, max_test_epoch=max_test_epoch, gamma=gamma,
                                  evaluate_every=evaluate_every, render=render)

        self.network = network
        # action bound
        self.action_bound = self.env.action_space.high
        # restore the model if possible
        self.restore()
        # evaluate the global step
        self.global_step = self.sess.run(self.step)
        # evaluate the global epoch
        self.global_epoch = self.sess.run(self.epoch)
        # check if training is done
        self.is_train = True if self.global_step < self.max_step else False
        # add summary
        self.q_summary = tf.scalar_summary("q_value", self.network.Q)

    def fit(self):
        # start training
        while self.global_step < self.max_step:
            if self.global_epoch % self.evaluate_every == 0:
                self.evaluate()
            curr_state = self.env.reset()
            curr_state = curr_state[np.newaxis]
            done = False
            self.policy.reset()
            while not done:
                if self.render:
                    self.env.render()

                if self.memory.count < self.warm_up:
                    action = self.env.action_space.sample()
                else:
                    # select action according to noise exploration
                    action = self.act(curr_state) + self.policy.noise() * self.action_bound

                next_state, reward, done, _ = self.env.step(action.flatten())
                next_state = next_state[np.newaxis]
                # if done, V' will not be considered
                terminal = 0 if done else 1
                # fill the replay buffer
                self.memory.add(curr_state, next_state, reward, terminal, action)

                # train the model after warm up
                if self.memory.count > self.warm_up and self.memory.count > self.memory.batch_size:
                    # update
                    for i in range(self.update_step):
                        # sample from memory
                        cs, ns, r, d, a = self.memory.sample()

                        # calculate y
                        y = r + self.gamma * d * self.target_value(ns).flatten()
                        # update network
                        loss, _ = self.sess.run([self.network.loss, self.network.update],
                                                feed_dict={self.network.y: y.reshape(self.batch_size, 1),
                                                           self.network.x: cs,
                                                           self.network.action: a.reshape(self.batch_size, -1)})
                        # update the target network
                        self.sess.run(self.network.target_update)

                curr_state = next_state

                # update step variable
                self.sess.run(self.step_assign)
                self.global_step += 1
            # update epoch variable
            self.sess.run(self.epoch_assign)
            self.global_epoch += 1

        self.is_train = False

        self.evaluate()

    def evaluate(self):
        pass

    def act(self, state):
        """
        Return action from the mu network
        """
        action = self.sess.run(self.network.mu, feed_dict={self.network.x: state})
        return action * self.action_bound

    def target_value(self, state):
        """
        Return V' from target network
        """
        V_prime = self.sess.run(self.network.target_V, feed_dict={self.network.target_x: state})
        return V_prime
