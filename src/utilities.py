import numpy as np


def progress(step, total_step, num_space):
    percentage = float(step)/total_step
    indicator = int(percentage * num_space)
    prog = ["["] + ["*" for _ in range(0, indicator)]
    prog += [" " for _ in range(0, num_space - indicator)] + ["]"]
    return ''.join(prog), percentage


def rgb2grey(image):
    """
    Convert RGB images to grey-scale
    """
    image = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.144
    return image/255

