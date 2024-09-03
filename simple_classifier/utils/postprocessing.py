import numpy as np


def softmax(x, temperature):
    """Compute softmax values with temperature for each of scores in x."""
    x /= temperature
    max_x = np.max(x, axis=-1)
    e_x = np.exp(x - max_x[:, None])
    return e_x / e_x.sum(axis=-1)[:, None]
