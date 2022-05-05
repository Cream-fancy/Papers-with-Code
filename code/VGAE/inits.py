import torch
import numpy as np
import torch.nn as nn


def glorot_init(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010) initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)
