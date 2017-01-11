from __future__ import division
from __future__ import print_function
from  __future__ import absolute_import
from src.nn_ops import *
import tensorflow as tf


class BaseNetwork(object):
    def __init__(self, input_dim, action_dim, update_option, name, optimizer, use_bn,
                 initializer=tf.contrib.layers.xavier_initializer(), tau=None):
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

        self.use_bn = use_bn
        self.update_option = update_option
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.initializer = initializer

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
    def __init__(self, input_dim, action_dim, tau, optimizer, use_bn=False, name="critic"):
        """
        Initialize critic network. The critic network maintains a copy of itself and target updating ops
        Args
            input_dim: dimension of input space, if is length one, we assume it is low dimension.
            action_dim: dimension of action space.
            stddev: standard deviation for initializing network params.
        """
        super(CriticNetwork, self).__init__(input_dim, action_dim, use_bn=use_bn, update_option="soft_update",
                                            name=name, optimizer=optimizer, tau=tau)

        self.update_op = self.create_update_op()

        if self.use_bn:
            self.network, self.x, self.action, self.network_train = self.network
            self.target, self.target_x, self.target_action, self.target_train = self.target
        else:
            self.network, self.x, self.action = self.network
            self.target, self.target_x, self.target_action = self.target

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=None, name="target_q")
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.network))

        # get gradients
        self.gradients, self.action_gradient = self.compute_gradient()

        # training operation
        self.train = self.create_train_op()

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.input_dim, name="%s_input" % name)
        action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="%s_action" % name)
        with tf.variable_scope(name):
            if len(self.input_dim) == 1:
                if self.use_bn:
                    is_train = tf.placeholder(tf.bool, name="%s_is_train" % name)

                    # normalize input
                    inpt = batch_norm(x, is_train=is_train, scope="norm_inpt")

                    # normalize first layer
                    net = dense_layer(inpt, 400, self.initializer, scope="fc1", use_bias=True)
                    net = tf.nn.relu(batch_norm(net, is_train, scope="fc1"))

                    # second layer without normalizing
                    net = tf.nn.relu(dense_layer(tf.concat(1, (net, action)), 300, use_bias=True, scope="fc2",
                                                 initializer=self.initializer))
                    net = dense_layer(net, 1, tf.random_uniform_initializer(-3e-3, 3e-3), scope="q", use_bias=True)
                    return tf.squeeze(net), x, action, is_train
                else:

                    net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
                    net = dense_layer(tf.concat(1, (net, action)), 300, use_bias=True, scope="fc2",
                                      initializer=self.initializer)
                    net = tf.nn.relu(net)

                    # for low dim, weights are from uniform[-3e-3, 3e-3]
                    net = dense_layer(net, 1, initializer=tf.random_uniform_initializer(-3e-3, 3e-3), scope="q",
                                      use_bias=True)
            else:
                # first convolutional layer with stride 4
                net = conv2d(x, 3, initializer=self.initializer, output_size=32, scope="conv1", stride=4, use_bias=True)
                net = tf.nn.relu(net)

                # second convolutional layer with stride 2
                net = conv2d(net, 3, stride=2, output_size=32, initializer=self.initializer, scope="conv2",
                             use_bias=True)
                net = tf.nn.relu(net)

                # third convolutional layer with stride 1
                net = conv2d(net, 3, stride=1, output_size=32, initializer=self.initializer, use_bias=True,
                             scope="conv3")
                net = tf.nn.relu(net)

                # first dense layer
                net = tf.nn.relu(dense_layer(net, output_dim=200, initializer=self.initializer, scope="fc1",
                                             use_bias=True))

                # second dense layer with action embedded
                net = tf.nn.relu(dense_layer(tf.concat(1, (net, action)), output_dim=200, initializer=self.initializer,
                                             scope="fc2", use_bias=True))

                # Q layer
                net = dense_layer(net, output_dim=1, initializer=tf.random_uniform_initializer(-4e-4, 4e-4), scope="Q",
                                  use_bias=True)
        return tf.squeeze(net), x, action

    def compute_gradient(self):
        grad = tf.gradients(self.loss, self.network_param, name="critic_gradients")
        action_gradient = tf.gradients(self.network, self.action, name="action_gradient")
        return grad, action_gradient


class ActorNetwork(BaseNetwork):
    def __init__(self, input_dim, action_dim, tau, optimizer, use_bn=False, name="actor"):
        """
        Initialize actor network
        """
        super(ActorNetwork, self).__init__(input_dim, action_dim, update_option="soft_update",
                                           name=name, optimizer=optimizer, tau=tau, use_bn=use_bn)

        self.update_op = self.create_update_op()

        if use_bn:
            self.network, self.x, self.network_train = self.network
            self.target, self.target_x, self.target_train = self.target
        else:
            self.network, self.x = self.network
            self.target, self.target_x = self.target

        # for actor network, we need to know the action gradient in critic network
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, action_dim), name="action_gradient")
        self.gradients = self.compute_gradient()
        self.train = self.create_train_op()

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.input_dim, name="%s_input" % name)
        with tf.variable_scope(name):
            if len(self.input_dim) == 1:
                # low dimensional case
                if self.use_bn:
                    is_train = tf.placeholder(tf.bool, name="%s_isTrain" % name)
                    # normalize input
                    inpt = batch_norm(x, is_train, "%s_norm_inpt" % name)

                    # normalize the first layer
                    net = dense_layer(inpt, 400, use_bias=True, scope="fc1", initializer=self.initializer)
                    net = tf.nn.relu(batch_norm(net, is_train=is_train, scope="fc1"))

                    # normalize the second layer
                    net = dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer)
                    net = tf.nn.relu(batch_norm(net, is_train=is_train, scope="fc2"))

                    # normalize the output layer
                    net = dense_layer(net, self.action_dim, use_bias=True,
                                      initializer=tf.random_uniform_initializer(-3e-3, 3e-3), scope="pi")
                    net = tf.nn.tanh(batch_norm(net, is_train=is_train, scope="pi"))

                    # return one more parameter: is_train
                    return net, x, is_train
                else:
                    net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
                    net = tf.nn.relu(dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer))
                    # use tanh to normalize output between [-1, 1]
                    net = tf.nn.tanh(dense_layer(net, self.action_dim,
                                                 initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                 scope="pi", use_bias=True))
            else:
                # first convolutional layer with stride 4
                net = conv2d(x, 3, initializer=self.initializer, output_size=32, scope="conv1", stride=4, use_bias=True)
                net = tf.nn.relu(net)

                # second convolutional layer with stride 2
                net = conv2d(net, 3, stride=2, output_size=32, initializer=self.initializer, scope="conv2",
                             use_bias=True)
                net = tf.nn.relu(net)

                # third convolutional layer with stride 1
                net = conv2d(net, 3, stride=1, output_size=32, initializer=self.initializer, use_bias=True,
                             scope="conv3")
                net = tf.nn.relu(net)

                # first dense layer
                net = tf.nn.relu(dense_layer(net, output_dim=200, initializer=self.initializer, scope="fc1",
                                             use_bias=True))

                # second dense layer with action embedded
                net = tf.nn.relu(dense_layer(net, output_dim=200, initializer=self.initializer,
                                             scope="fc2", use_bias=True))
                # Q layer
                net = tf.tanh(dense_layer(net, output_dim=self.action_dim,
                                          initializer=tf.random_uniform_initializer(-4e-4, 4e-4),
                                          scope="pi", use_bias=True))
            return net, x

    def compute_gradient(self):
        # We negate action gradient because we want the parameters
        # follow the direction of the performance gradients. Optimizer
        # will apply gradient descent, but we want to ascent the gradient.

        grads = tf.gradients(self.network, self.network_param, -self.action_gradient)
        return grads
