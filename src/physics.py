import numpy as np


class IsingModel:
    def __init__(self, temperature=300):
        self.temperature = temperature
        self.dimensionless_temperature = temperature
    
    def hamiltonian(self, state):
        return - nearest_neighbour_sum(state, len(state.shape))
    
    def probability_distribution(self, state):
        return np.exp(-self.hamiltonian(state)/self.dimensionless_temperature)


def nearest_neighbour_sum(state, dimensions):
    neighbours = np.zeros(shape=(2*dimensions, *state.shape), dtype=np.bool_)

    n, m = state.shape
    roll_backwards_idx = np.arange(-1, n-1)
    roll_forward_idx = np.arange(1-n, 1)

    for dimension in range(dimensions):
        neighbours[2*dimension  ,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_forward_idx], 0, dimension)
        neighbours[2*dimension+1,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_backwards_idx], 0, dimension)

    return np.sum(state*neighbours)