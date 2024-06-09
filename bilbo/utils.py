import numpy as np


def rand_between(a: float, b: float) -> float:
    return np.random.rand() * (b - a) + a
