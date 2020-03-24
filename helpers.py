import numpy as np


def normalized_dot_product(u, w):
    z = u * np.dot(u, w) / np.dot(u, u)
    return z