from gym import Wrapper
from src.utilities import rgb2grey
import numpy as np
import cv2


class NormalizeWrapper(Wrapper):
    def __init__(self, env, min_value, max_value):
        """
        Create a wrapper for OpenAI Gym environment. This wrapper normalizes the game state between
        min_value and max_value
        """
        super(NormalizeWrapper, self).__init__(env)
        self.min_value = min_value
        self.max_value = max_value
        self.env_min_value = self.observation_space.low
        self.env_max_value = self.observation_space.high

    def _step(self, action):
        """
        Take an action and return the normalized state values
        """
        obs, r, d, info = self.env.step(action)
        obs = self.__normalize(obs)
        return obs, r, d, info

    def _reset(self):
        obs = self.env.reset()
        return self.__normalize(obs)

    def __normalize(self, state):
        """
        Normalize the state values between min_value and max_value. (min_value and max_value could be numpy arrays)
        """
        normalized_state = state.flatten()
        normalized_state = (self.max_value - self.min_value) * (normalized_state - self.env_min_value) / \
                           (self.env_max_value - self.env_min_value) + self.min_value

        return normalized_state


class FrameSkippingWrapper(Wrapper):
    def __init__(self, env, num_skipping=4, height=84, width=84):
        """
        Create the frame skipping wrapper. Normalize each frame into gray-scaled image and stack continuous
        n frames together.
        """
        super(FrameSkippingWrapper, self).__init__(env)

        self.height = height
        self.width = width
        self.num_skipping = num_skipping
        self.history = np.empty((height, width, self.num_skipping))

    def _step(self, action):
        """
        Override parent's _step function to stack n images together.
        Return a (84, 84, 4) state tensor, action repeat for n times.
        """
        s, r, done, info = self.env.step(action.flatten())
        self.history[..., :-1] = self.history[..., 1:]
        self.history[..., -1] = cv2.resize(rgb2grey(s), (self.width, self.height))
        return self.history, r, done, info

    def _reset(self):
        obs = self.env.reset()
        for i in range(self.num_skipping):
            self.history[..., i] = cv2.resize(rgb2grey(obs), (self.width, self.height))
        return self.history


if __name__ == '__main__':
    import gym
    from matplotlib import pyplot as plt
    env = gym.make("Breakout-v0")
    env = FrameSkippingWrapper(env, 4)
    obs = env.reset()
    while True:
        obs, r, done, info = env.step(env.action_space.sample())
        if done:
            break
    plt.subplot(4, 1, 1)
    plt.imshow(obs[..., 0], cmap='Greys')
    plt.subplot(4, 1, 2)
    plt.imshow(obs[..., 1], cmap='Greys')
    plt.subplot(4, 1, 3)
    plt.imshow(obs[..., 2], cmap='Greys')
    plt.subplot(4, 1, 4)
    plt.imshow(obs[..., 3], cmap='Greys')
    plt.show()
