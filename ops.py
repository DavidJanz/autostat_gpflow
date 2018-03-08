import numpy as np


def BIC(log_likelihood, n_params, n_data):
    return np.log(n_data) * n_params - 2 * log_likelihood