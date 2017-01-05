import numpy as np


def progress(step, total_step, num_space):
    """

    :param step:
    :param total_step:
    :param num_space:
    :return:
    """
    percentage = float(step)/total_step
    indicator = int(percentage * num_space)
    prog = ["["] + ["*" for _ in range(0, indicator)]
    prog += [" " for _ in range(0, num_space - indicator)] + ["]"]
    return ''.join(prog), percentage


class Normalizer(object):
    def __init__(self, low, high, norm_min, norm_high):
        self.low = low
        self.high = high
        self.norm_min = norm_min
        self.norm_high = norm_high

    def normalize(self, value):

        x = value.flatten()
        x = (self.norm_high - self.norm_min) * ((x - self.low) / (self.high - self.low)) + self.norm_min
        return x
