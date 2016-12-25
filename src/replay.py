from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np


class Memory(object):
    """
    An implementation of the replay memory. This is essential when dealing with DRL algorithms that are not
    multi-threaded as in A3C.
    """

    def __init__(self, memory_size, save_dir, state_dim, action_dim, batch_size):
        """
        A naive implementation of the replay memory, need to do more work on this after testing DDPG
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.save_dir = save_dir

        if type(state_dim) is not tuple:
            state_dim = (state_dim, )

        # current state
        self.curr_state = np.empty(shape=(memory_size, ) + state_dim)
        # next state
        self.next_state = np.empty(shape=(memory_size, ) + state_dim)
        # reward
        self.rewards = np.empty(memory_size)
        # terminal
        self.terminals = np.empty(memory_size)
        # actions
        self.actions = np.empty((memory_size, action_dim) if action_dim > 1 else memory_size)

        self.current = 0
        self.count = 0

    def add(self, curr_state, next_state, reward, terminal, action):
        self.curr_state[self.current, ...] = curr_state
        self.next_state[self.current, ...] = next_state
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.actions[self.current] = action

        self.current += 1
        self.count = max(self.count, self.current)
        if self.current >= self.memory_size - 1:
            self.current = 0

    def sample(self):
        indexes = np.random.randint(0, self.count, self.batch_size)

        curr_state = self.curr_state[indexes, ...]
        next_state = self.next_state[indexes, ...]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        actions = self.actions[indexes]

        return curr_state, next_state, rewards, terminals, actions

if __name__ == '__main__':
    current_state, next_state = np.random.randn(1, 2), np.random.randn(1, 2)
    print(current_state)
    print(next_state)
    memory = Memory(3, "", 2, 1, 1)
    memory.add(current_state, next_state, 1.5, False, 2)
    memory.add(current_state, next_state, 1.2, False, 1)
    print(memory.sample())
    print(memory.sample())
