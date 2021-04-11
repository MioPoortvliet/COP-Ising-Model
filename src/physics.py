import numpy as np
from numba import njit
from itertools import product

# True is down
# False is up!!!!!

class IsingModel:
    def __init__(self, temperature=300, dims=2):
        self.temperature = temperature
        self.dimensionless_temperature = temperature
        self.energy_difference_nn_lookup = generate_energy_difference_lookup(self.energy_difference_nearest_neighbours, dims)
        self.dims = dims

        eyes = np.eye(self.dims, dtype=np.int64)
        self.coordinate_deltas = np.concatenate((eyes, -eyes))
    
    def hamiltonian(self, state):
        return - nearest_neighbour_sum(state, len(state.shape))

    def hamiltonian_to_probability(self, hamiltonian):
        return np.exp(-hamiltonian / self.dimensionless_temperature)
    
    def probability_distribution(self, state):
        return np.exp(-self.hamiltonian(state)/self.dimensionless_temperature)

    def get_neighbour_spins(self, coordinate, state):
        neighbour_spins = np.zeros(2**self.dims, dtype=np.bool)

        for i, coords in enumerate(np.mod(np.array(coordinate)+self.coordinate_deltas, state.shape)):
            neighbour_spins[i] = state[tuple(coords)]

        return neighbour_spins

    def energy_difference(self, coordinate, state):
        neighbour_spins = self.get_neighbour_spins(coordinate, state)
        index = tuple(np.array([state[coordinate], *neighbour_spins]).astype(int))

        return self.energy_difference_nn_lookup[index]

    def energy_difference_old(self, coordinate, state):
        energy_difference = 0
        for dimension in range(len(state.shape)):
            offset = np.zeros(len(state.shape), dtype=np.int)
            offset[dimension] = 1
            energy_difference -= bool_to_spin(np.logical_xor(state[coordinate], state[tuple(np.mod(coordinate+offset, state.shape))]))
            energy_difference -= bool_to_spin(np.logical_xor(state[coordinate], state[tuple(np.mod(coordinate-offset, state.shape))]))

        return energy_difference * 2

    def energy_difference_nearest_neighbours(self, own_spin, neighbors_spin):
        energy_difference = 0
        for spin in neighbors_spin:
            energy_difference -= bool_to_spin(np.logical_xor(own_spin, spin))

        return energy_difference * 2


@njit
def bool_to_spin(state):
    return state * -2 + 1


def generate_energy_difference_lookup(func, dims):
    lookup = np.zeros(shape=np.repeat(2, 2*dims+1), dtype=np.int)

    for bools in product([True, False], repeat=2*dims+1):
        index = tuple(np.array(bools).astype(int))
        #print(index)
        lookup[index] = func(bools[0], bools[1:])
        #print(lookup[index], func(bools[0], bools[1:]))
        #print()

    return lookup


def nearest_neighbour_sum(state, dimensions):
    neighbours = np.zeros(shape=(2*dimensions, *state.shape), dtype=np.bool_)

    n, m = state.shape
    roll_backwards_idx = np.arange(-1, n-1)
    roll_forward_idx = np.arange(1-n, 1)

    # Loop through dims and find neighbour states to every spin
    for dimension in range(dimensions):
        neighbours[2*dimension  ,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_forward_idx], 0, dimension)
        neighbours[2*dimension+1,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_backwards_idx], 0, dimension)

    return np.sum(bool_to_spin(np.logical_xor(state, neighbours)))


def magnetization(state):
    return np.sum(bool_to_spin(state)) / state.size
