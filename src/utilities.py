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
