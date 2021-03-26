import numpy as np
from numba import njit

class IsingModel:
    def __init__(self, temperature=300):
        self.temperature = temperature
        self.dimensionless_temperature = temperature
    
    def hamiltonian(self, state):
        return - nearest_neighbour_sum(state, len(state.shape))
    
    def probability_distribution(self, state):
        return np.exp(-self.hamiltonian(state)/self.dimensionless_temperature)
    
#@njit
def nearest_neighbour_sum(state, dimensions):
    neighbours = np.zeros(shape=(2*dimensions, *state.shape), dtype=bool)
    
    for dimension in np.arange(dimensions):
        neighbours[2*dimension   ,:,:] = np.roll(state, 1, axis=dimension)
        neighbours[2*dimension +1,:,:] = np.roll(state, -1, axis=dimension)

    return np.sum(state*neighbours)