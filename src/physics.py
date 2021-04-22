"""
Contains all physics of the Ising model system.
Authors: Mio Poortvliet, Jonah Post
"""
import numpy as np
from numba import njit
from itertools import product
from typing import Callable

# True is down
# False is up!!!!!

class IsingModel:
    def __init__(self, dimensionless_temperature=1., dims=2) -> None:
        """
        Initializes the physics of the system. This class contains functions about the energy.
        :param dimensionless_temperature: Temperature * kB / J
        :type dimensionless_temperature: float
        :param dims: dimensions of system
        :type dims: int
        """
        # Only take dimensionless temperature
        self.dimensionless_temperature = dimensionless_temperature
        self.dims = dims
        # Generate a lookup table to speed up calculations significantly
        self.energy_difference_nn_lookup = generate_energy_difference_lookup(energy_difference_nearest_neighbours, dims)

        # Temporary variable
        eyes = np.eye(self.dims, dtype=np.int64)
        # We use this array to find the nearest neighbours later.
        self.coordinate_deltas = np.concatenate((eyes, -eyes))
    
    def hamiltonian(self, state:np.ndarray) -> int:
        """
        Returns the hamiltonian (J=1). As a consequence it is an int.
        """
        return - nearest_neighbour_sum(state, len(state.shape))

    def hamiltonian_to_probability(self, hamiltonian:int) -> float:
        """Returns the probability of this energy configuration existing, given hamiltonian"""
        return np.exp(-hamiltonian / self.dimensionless_temperature)
    
    def probability_distribution(self, state:np.ndarray) -> float:
        """Returns the probability of this energy configuration existing, given state """
        return np.exp(-self.hamiltonian(state)/self.dimensionless_temperature)

    def get_neighbour_spins(self, coordinate:tuple, state:np.ndarray) -> np.ndarray:
        """Returns the spins around coordinates (any dimensionality!). Takes `infinite plane' into account."""
        coordinates = np.mod(np.array(coordinate)+self.coordinate_deltas, state.shape)
        neighbour_spins = state[tuple([coordinates[::, d] for d in range(self.dims)])]

        return neighbour_spins

    def energy_difference(self, coordinate:tuple, state:np.ndarray) -> int:
        """Returns energy difference on hamiltonian given coordinate and state.
        If the spin at coordinate is flipped, how does energy change? This is what it returns."""
        # Find neighbour spins
        neighbour_spins = self.get_neighbour_spins(coordinate, state)
        # Use this to get the index of the lookup table
        # Stupid, but we need this specific conversion. I'm sure this makes it slower but idk.
        index = tuple(np.array([state[coordinate], *neighbour_spins]).astype(int))

        return self.energy_difference_nn_lookup[index]

    def energy_difference_nearest_neighbours(self, own_spin:bool, neighbours_spin:np.ndarray) -> np.int:
        """
        Given the nearest neighbour spins, what is the energy difference?
        :type own_spin: bool
        :type neighbours_spin: np.ndarray, dtype=np.bool
        :return: energy difference in unitless units
        :rtype: int (could be float in general, not in our case!)
        """
        energy_difference = 0
        for spin in neighbours_spin:
            # Mulitplication of 1 and -1 works like an xor
            energy_difference -= bool_to_spin(np.logical_xor(own_spin, spin))


        return energy_difference * 2

    def energy_difference_old(self, coordinate, state):
        """
        DEPRECATED
        """
        energy_difference = 0
        for dimension in range(len(state.shape)):
            offset = np.zeros(len(state.shape), dtype=np.int)
            offset[dimension] = 1

            # Mulitplication of 1 and -1 works like an xor
            energy_difference -= bool_to_spin(np.logical_xor(state[coordinate], state[tuple(np.mod(coordinate+offset, state.shape))]))
            energy_difference -= bool_to_spin(np.logical_xor(state[coordinate], state[tuple(np.mod(coordinate-offset, state.shape))]))

        return energy_difference * 2

# jit-compile using numba for speedup
# It is called in magnetization too!
@njit
def bool_to_spin(state) -> int:
    """Bool to spin conversion."""
    return state * -2 + 1


@njit
def magnetization(state: np.ndarray) -> np.ndarray:
    """
    Calculate magnetization given a state. Performs spin conversion, then sums.
    :param state: state array
    :return: magnetization
    """
    return np.sum(bool_to_spin(state))


@njit
def energy_difference_nearest_neighbours(own_spin:bool, neighbours_spin:np.ndarray) -> np.int:
    """
    Given the nearest neighbour spins, what is the energy difference?
    :type own_spin: bool
    :type neighbours_spin: np.ndarray, dtype=np.bool
    :return: energy difference in unitless units
    :rtype: int (could be float in general, not in our case!)
    """
    energy_difference = 0
    for spin in neighbours_spin:
        # Mulitplication of 1 and -1 works like an xor
        energy_difference -= bool_to_spin(np.logical_xor(own_spin, spin))

    return energy_difference * 2


def generate_energy_difference_lookup(func:Callable[[bool, np.ndarray],int], dims:int) -> np.ndarray:
    """Create a lookup table of the function for the cartesian product of the set {true, false}^(2*dims+1)
    Assumes the function takes 2*dims+1 boolean arguments (2*dims nearest neighbours and one own spin)."""
    lookup = np.zeros(shape=np.repeat(2, 2*dims+1), dtype=np.int)

    # Try all combinations now and save them.
    for bools in product([True, False], repeat=2*dims+1):
        # We index using booleans, but they are not a mask! That is why they need to be a tuple of ints.
        index = tuple(np.array(bools).astype(int))
        lookup[index] = func(bools[0], bools[1:])

    return lookup


def nearest_neighbour_sum(state:np.ndarray, dimensions:int) -> np.ndarray:
    """Perform the sum of spins over nearest neighbours, S_i S_j.
    Returns the sum of spins of the product of all nearest neighbours pairs."""
    # Initialize empety array
    neighbours = np.zeros(shape=(2*dimensions, *state.shape), dtype=np.bool_)

    # Indices to roll the array
    # Grid needs to be square due to this!
    roll_backwards_idx = np.arange(-1, state.shape[0]-1)
    roll_forward_idx = np.arange(1-state.shape[0], 1)

    # Loop through dims and find neighbour states to every spin
    for dimension in range(dimensions):
        # We swap axes to get the right number in the right place. This is because we want it to work for n dims.
        neighbours[2*dimension  ,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_forward_idx], 0, dimension)
        neighbours[2*dimension+1,:,:] = np.swapaxes(np.swapaxes(state, dimension, 0)[roll_backwards_idx], 0, dimension)

    # The xor works like spin multiplication (just write down a truth table and it's obvious)
    return np.sum(bool_to_spin(np.logical_xor(state, neighbours))) / 2