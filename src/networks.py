from __future__ import division
from __future__ import print_function
from nn_ops import *
import tensorflow as tf


class BaseNetwork(object):
    def __init__(self, input_dim, action_dim, stddev, update_option, name, optimizer, tau=None):
        """
        Abstarct class for creating networks
        :param input_dim:
        :param action_dim:
        :param stddev:
        """

        # if use soft update, tau should not be None
        if update_option == "soft_update":
            assert (tau is not None), "Soft update needs to specify tau"
            self.tau = tau

        self.update_option = update_option
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.initializer = tf.truncated_normal_initializer(stddev=stddev)

        # build network
        self.network = self.build(name)
        self.network_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                              "target" not in v.name]

        # build target
        self.target = self.build("target_%s" % name)
        self.target_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                             "target" in v.name]

        self.gradients = None

        # optimizer
        self.optimizer = optimizer

    def create_update_op(self):
        if self.update_option == "soft_update":
            update_op = [tf.assign(target_param, (1 - self.tau) * target_param + self.tau * network_param)
                         for target_param, network_param in zip(self.target_param, self.network_param)]
        else:
            update_op = [tf.assign(target_param, network_param)
                         for target_param, network_param in zip(self.target_param, self.network_param)]

        return update_op

    def create_train_op(self):
        return self.optimizer.apply_gradients([(g, v) for g, v in zip(self.gradients, self.network_param)])

    def build(self, name):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def compute_gradient(self):
        """
        Abstract method, compute gradient in order to be used by self.optimizer
        """
        raise NotImplementedError("Not implemented")


class CriticNetwork(BaseNetwork):
    def __init__(self, input_dim, action_dim, tau, stddev, optimizer, name="critic"):
        """
        Initialize critic network. The critic network maintains a copy of itself and target updating ops
        Args
            input_dim: dimension of input space, if is length one, we assume it is low dimension.
            action_dim: dimension of action space.
            stddev: standard deviation for initializing network params.
        """
        super(CriticNetwork, self).__init__(input_dim, action_dim, update_option="soft_update",
                                            name=name, stddev=stddev, optimizer=optimizer, tau=tau)

        self.update_op = self.create_update_op()
        self.network, self.x, self.action = self.network
        self.target, self.target_x, self.target_action = self.target
        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=(None, 1), name="target_q")
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.network))

        # get gradients
        self.gradients, self.action_gradient = self.compute_gradient()

        # training operation
        self.train = self.create_train_op()

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_dim, name="%s_input" % name)
        action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="%s_action" % name)
        with tf.variable_scope(name):
            if len(self.input_dim) == 1:
                net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
                x1 = dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer)
                # include action
                x2 = dense_layer(action, 300, use_bias=True, scope="action_embedding",
                                 initializer=self.initializer)
                net = tf.nn.relu(x1 + x2)

                # for low dim, weights are from uniform[-3e-3, 3e-3]
                net = dense_layer(net, 1, initializer=tf.random_uniform_initializer(-3e-3, 3e-3), scope="q",
                                  use_bias=True)
            else:
                pass
            # net = dense_layer(net, 1, initializer=self.initializer, scope="q", use_bias=True)
        return net, x, action

    def compute_gradient(self):
        grad = tf.gradients(self.loss, self.network_param, name="critic_gradients")
        action_gradient = tf.gradients(self.network, self.action, name="action_gradient")
        return grad, action_gradient


class ActorNetwork(BaseNetwork):
    def __init__(self, input_dim, action_dim, tau, stddev, optimizer, name="actor"):
        """
        Initialize actor network
        """
        super(ActorNetwork, self).__init__(input_dim, action_dim, update_option="soft_update",
                                           name=name, stddev=stddev, optimizer=optimizer, tau=tau)

        self.update_op = self.create_update_op()
        self.network, self.x = self.network
        self.target, self.target_x = self.target

        # for actor network, we need to know the action gradient in critic network
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, action_dim), name="action_gradient")
        self.gradients = self.compute_gradient()
        self.train_op = self.create_train_op()

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_dim, name="%s_input" % name)
        with tf.variable_scope(name):
            if len(self.input_dim) == 1:
                net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
                net = tf.nn.relu(dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer))

                # use tanh to normalize output between [-1, 1]
                net = tf.nn.tanh(dense_layer(net, self.action_dim,
                                             initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                             scope="pi", use_bias=True))
            else:
                pass

        return net, x

    def compute_gradient(self):
        # We negate action gradient because we want the parameters
        # follow the direction of the performance gradients. Optimizer
        # will apply gradient descent, but we want to ascent the gradient.

        grads = tf.gradients(self.network, self.network_param, -self.action_gradient)
        return grads