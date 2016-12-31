import tensorflow as tf
import os


class BaseAgent(object):
    def __init__(self, sess, memory, env, env_name, max_step, normalizer=None, warm_up=5000, max_test_epoch=10, render=True, gamma=.99):
        """
        Base agent. Provide basic functions: save, restore, perform training and evaluation (abstract method).
        """
        self.sess = sess

        self.memory = memory
        self.batch_size = memory.batch_size

        self.max_test_epoch = max_test_epoch
        self.env = env
        self.env_name = env_name
        self.warm_up = warm_up
        self.normalizer = normalizer

        self.render = render
        self.gamma = gamma
        self.max_step = max_step

        with tf.variable_scope("step"):
            self.step = tf.Variable(0, trainable=False)
            # increase the step every time one training iteration is completed.
            self.step_assign = tf.assign_add(self.step, tf.Variable(1, trainable=False))

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
            print("Successfully load the model %s" % ckpt)
        else:
            print("Model Restore Failed %s" % self.save_dir)

    def fit(self, max_step_per_game=None):
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

