from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf


class ReplayMemory(object):
    """
    An implementation of the replay memory. This is essential when dealing with DRL algorithms that are not
    multi-threaded as in A3C.
    """
