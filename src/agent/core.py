import tensorflow as tf
import os
from gym.wrappers import Monitor


class BaseAgent(object):
    def __init__(self, sess, memory, env, env_name, max_step,
                 evaluate_every=10, warm_up=5000, max_test_epoch=10, render=True, gamma=.99):
        """
        Base agent. Provide basic functions: save, restore, perform training and evaluation (abstract method).
        Args:
            sess: tf.Session() variable
            memory: Replay Memory
            env: OpenAI Gym environment or a wrapped environment
            env_name: a string indicating the name of the env
            max_step: maximum step to perform training
            evaluate_every: how many episode to evaluate the current policy
            warm_up: how many steps to take on random policy
            max_test_epoch: maximum epochs to evaluate the policy after finishing training
            render: if show the training process
            gamma: discount factor
        """
        self.sess = sess

        self.memory = memory
        self.batch_size = memory.batch_size

        self.max_test_epoch = max_test_epoch
        self.env_name = env_name
        self.warm_up = warm_up
        self.evaluate_every = evaluate_every
        self.monitor_dir = os.path.join("tmp", type(self).__name__, env_name)
        self.monitor = Monitor(env, directory=os.path.join(self.monitor_dir, "train"),  #TODO: when episode_id > 100, this will not be in the evaluation.
                               video_callable=lambda x: (x - x / self.evaluate_every) % self.evaluate_every == 0)
        self.env = env

        # for summary writer
        self.logdir = os.path.join("log", type(self).__name__, env_name)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # create summary writer and summary object
        self.writer = tf.summary.FileWriter(self.logdir, graph=self.sess.graph)
        self.summary = tf.Summary()

        self.render = render
        self.gamma = gamma
        self.max_step = max_step

        with tf.variable_scope("step"):
            self.step = tf.Variable(0, trainable=False)
            # increase the step every time one training iteration is completed.
            self.step_assign = tf.assign_add(self.step, tf.Variable(1, trainable=False))
            self.epoch = tf.Variable(0, trainable=False, name="global_epoch")
            self.epoch_assign = tf.assign_add(self.epoch, tf.Variable(1, trainable=False))

        # save directory
        self.save_dir = os.path.join("models", env_name)
        # check if the save directory is there
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.sess.run(tf.global_variables_initializer())

        # create saver
        self.saver = tf.train.Saver()

    def save(self):
        """
        Save all model parameters and replay memory to self.save_dir folder.
        The save_path should be models/env_name/name_of_agent.
        """
        # path to the checkpoint name
        path = os.path.join(self.save_dir, type(self).__name__)
        print("Saving the model to path %s" % path)
        self.memory.save(self.save_dir)
        self.saver.save(self.sess, path)

    def restore(self):
        """
        Restore model parameters and replay memory from self.save_dir folder.
        The name of the folder should be models/env_name
        """
        ckpts = tf.train.get_checkpoint_state(self.save_dir)
        if ckpts and ckpts.model_checkpoint_path:
            ckpt = ckpts.model_checkpoint_path
            self.saver.restore(self.sess, ckpt)
            self.memory.restore(self.save_dir)
            print("Successfully load the model %s" % ckpt)
            print("Memory size is:")
            self.memory.size()
        else:
            print("Model Restore Failed %s" % self.save_dir)

    def fit(self):
        """
        Train the model. The agent training process will end at self.max_step.
        If max_step_per_game is provided, the agent will perform a limited steps
        during each game.
        """
        raise NotImplementedError("This method should be implemented")

    def evaluate(self):
        """
        Evaluate the model. This should only be called when self.max_epoch is reached.
        The evaluation will be recorded.
        """
        raise NotImplementedError("This method should be implemented")

