import numpy as np
from numba import njit

# True is down
# False is up!!!!!

class IsingModel:
    def __init__(self, temperature=300):
        self.temperature = temperature
        self.dimensionless_temperature = temperature
    
    def hamiltonian(self, state):
        return - nearest_neighbour_sum(state, len(state.shape))

    def hamiltonian_to_probability(self, hamiltonian):
        return np.exp(-hamiltonian / self.dimensionless_temperature)
    
    def probability_distribution(self, state):
        return np.exp(-self.hamiltonian(state)/self.dimensionless_temperature)

    # @mio vetorize this!
    def energy_difference(self, coordinate, state):
        energy_difference = 0
        for dimension in range(len(state.shape)):
            offset = np.zeros(len(state.shape), dtype=np.int)
            offset[dimension] = 1
            energy_difference -= spin(np.logical_xor(state[coordinate], state[tuple(np.mod(coordinate+offset, state.shape))]))
            energy_difference -= spin(np.logical_xor(state[coordinate], state[tuple(np.mod(coordinate-offset, state.shape))]))

        return energy_difference * 2

@njit
def spin(state):
    return state * -2 + 1


def nearest_neighbour_sum(state, dimensions):
    neighbours = np.zeros(shape=(2*dimensions, *state.shape), dtype=np.bool_)

    n, m = state.shape
    roll_backwards_idx = np.arange(-1, n-1)
    roll_forward_idx = np.arange(1-n, 1)

    # Loop through dims and find neighbour states to every spin
    for dimension in range(dimensions):
        neighbours[2*dimension  ,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_forward_idx], 0, dimension)
        neighbours[2*dimension+1,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_backwards_idx], 0, dimension)

    return np.sum(spin(np.logical_xor(state, neighbours)))


def magnetization(state):
    return np.sum(spin(state)) / state.size
