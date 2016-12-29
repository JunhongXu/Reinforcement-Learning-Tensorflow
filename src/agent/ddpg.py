from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from src.agent.core import BaseAgent
import numpy as np
from src.utilities import progress
import tensorflow as tf

# TODO: 1. Add summary to max Q value, average Q value, and total reward for one episode.
# TODO: 2. Modify reply memory so it is able to store and sample high dimensional data.
# TODO: 3. Add batch normalization to actor and critic networks
# TODO: 4. Add high dimensional network build.
# TODO: 5. Create functions to save replay memory (in Memory class).
# TODO: 6. Create testing function (in DDPG)
# TODO: 7. Create Policy class so that different policies can be implemented.

class DDPG(BaseAgent):
    def __init__(self, sess, critic, actor, env, env_name, memory, max_step, gamma=.99, render=True):
        """
        A deep deterministic policy gradient agent.
        """
        super(DDPG, self).__init__(sess, env=env, memory=memory, gamma=gamma, render=render,max_step=max_step,
                                   env_name=env_name)

        # action bound
        self.action_bound = env.action_space.high

        # critic network
        self.critic = critic

        # actor network
        self.actor = actor

        self.restore()

    def fit(self, max_step_per_game=None):
        # evaluate step variable
        step = self.sess.run(self.step)

        print("Starting from step %s" % step)
        # iterate until reach the maximum step
        while step < self.max_step:
            current_state = self.env.reset()
            per_game_reward = 0
            done = False
            print("Progress: %s %s" % (progress(step, self.max_step, 50)[0],progress(step, self.max_step, 50)[1]))
            while not done:
                if self.render:
                    self.env.render()
                current_state = current_state.reshape(1, -1)
                action = self.actor_predict(current_state) + np.random.normal(1/(step+1.0), size=self.actor.action_dim)
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, -1)
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
                    # r = np.tanh(r)

                    # step 1: target action
                    target_action = self.actor_target_predict(next_s)

                    # step 2: estimate next state's Q value according to this action (target)
                    y = self.critic_target_predict(next_s, target_action)
                    y = r + t * self.gamma * y

                    # step 3: update critic
                    loss, _ = self.sess.run([self.critic.loss, self.critic.train],
                                            feed_dict={self.critic.y: y, self.critic.action: a, self.critic.x: s})
                    # step 4: perceive action according to actor given s
                    actor_action = self.actor_predict(s)

                    # step 5: calculate action gradient
                    action_gradient = self.sess.run(self.critic.action_gradient,
                                                    {self.critic.x: s, self.critic.action: actor_action})
                    # step 6: update actor policy
                    self.sess.run(self.actor.train, {self.actor.x: s, self.actor.action_gradient: action_gradient[0]})

                    # update targets
                    self.sess.run([self.actor.update_op, self.critic.update_op])
                    # end of training

                if step % 400 == 0:
                    self.save()

                current_state = next_state

                # if max step is provided, terminate at the max_step during every episode.
                if step % max_step_per_game == 0 and step != 0:
                    done = True

                # increase step and step variable
                step += 1
                self.sess.run(self.step_assign)

            print("At step %s, total reward is %s" % (step, per_game_reward))

    def evaluate(self, max_test_epoch):
        pass

    def critic_predict(self, state, action):
        return self.sess.run(self.critic.network,
                             feed_dict={self.critic.action: action, self.critic.x: state})

    def critic_target_predict(self, state, action):
        return self.sess.run(self.critic.target,
                             feed_dict={self.critic.target_action: action, self.critic.target_x: state})

    def actor_predict(self, state):
        return self.sess.run(self.actor.network,
                             feed_dict={self.actor.x: state}) * self.action_bound

    def actor_target_predict(self, state):
        return self.sess.run(self.actor.target,
                             feed_dict={self.actor.target_x: state}) * self.action_bound
