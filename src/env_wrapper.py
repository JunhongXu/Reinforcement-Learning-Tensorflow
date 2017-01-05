from gym import Wrapper


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

