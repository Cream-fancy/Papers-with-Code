import torch as th
import numpy as np
import torch.nn as nn


def uniform(shape, scale=0.05):
    """Uniform init."""
    initial = th.rand(shape, dtype=th.float32) * 2 * scale - scale
    return nn.Parameter(initial)


def glorot(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010) initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = th.rand(input_dim, output_dim, dtype=th.float32) * 2 * init_range - init_range
    return nn.Parameter(initial)


def zeros(shape):
    """All zeros."""
    initial = th.zeros(shape, dtype=th.float32)
    return nn.Parameter(initial)


def ones(shape):
    """All ones."""
    initial = th.ones(shape, dtype=th.float32)
    return nn.Parameter(initial)
