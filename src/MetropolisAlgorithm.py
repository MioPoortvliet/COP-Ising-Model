import numpy as np
from src.utils import choice, Nchoice


class MetropolisAlgorithm:
    """Implementation of the Metropolis Algorithm as described in the lecture script."""

    def __init__(self, model:object, property_functions:tuple, settings:dict) -> None:
        """
        Initialization method. Model describes the system.

        :param model: needs methods: model.hamiltonian, model.energy_difference, model.hamiltonian_to_probability
        :type model: object
        :param property_functions: Functions to calculate every step. Must take self.state as only argument.
        :type property_functions: tuple
        :param settings: settings such as dimension, (particles along an axis) and initial distribution.
        :type settings: dict
        """

        # Set up
        self.model = model
        self.propty_functions = property_functions
        self.settings = settings
        self.size = settings["size"]
        self.dimensions = settings["dimensions"]
        self.initial_distribution = settings["initial_distribution"]

        # Create state
        self.state = Nchoice(N=self.size ** self.dimensions, p=self.initial_distribution)
        # Below is SLOW!
        #self.state = np.random.choice(a=[True, False], size=self.size**self.dimensions, p=[self.initial_distribution,1-self.initial_distribution])

        # Give the array the proper shape, need np.repeat to match variable number of dimensions
        self.state = np.reshape(self.state, newshape=np.repeat(self.size, self.dimensions))

        self.total_spins = self.state.size

        # Need to calculate this once, add energy differences in the future
        self.current_energy = self.model.hamiltonian(self.state)


    def make_new_state(self) -> tuple:
        """
        Modifies self.new_state inplace at coordinates returned.
        :return: random coordinates
        :rtype: tuple
        """
        # Draw a random coordinate
        coordinate = tuple(np.random.randint(0,self.size,self.dimensions))

        # We create self.new_state here and use it later. Sloppy, I know.
        self.new_state = np.copy(self.state)
        # Flip spin at coordinate
        self.new_state[coordinate] = not self.new_state[coordinate]

        return coordinate


    def step(self) -> None:
        """
        Propegate the simulation a step forward. Does not calculate any property functions.
        """
        # Do a walk in state space
        coordinate = self.make_new_state()
        # Metropolis Algorithm
        energy_difference = self.model.energy_difference(coordinate, self.new_state)

        if energy_difference < 0:
            acceptance_probability = 1
        else:
            acceptance_probability = self.model.hamiltonian_to_probability(energy_difference)

        # Too slow:
        #if np.random.choice(a=[True, False] , p=[acceptance_probability,1-acceptance_probability]):

        # We use this
        if choice(p=acceptance_probability):
            # Accept state if true
            self.state = np.copy( self.new_state)
            self.current_energy += energy_difference


    def run_steps(self, steps:int) -> np.ndarray:
        """
        Runs the simulation for steps. Does calculate property functions, returns it.
        :param steps: steps to propegate simulation
        :type steps: int
        :return: array of saved properties
        :rtype: np.ndarray
        """
        # We define this here for the first time, but we need to call it in calc_properties and don't need it outside this function.
        self.saved_properties = np.zeros(shape=(steps+1, len(self.propty_functions)+1))
        # Make sure we don't miss anything
        self.calc_properties(0)

        for i in range(steps):
            self.step()
            self.calc_properties(i+1)

        return self.saved_properties

    def calc_properties(self, i=0) -> None:
        """
        Calculates and write the properties in array self.saved_properties at step i.
        :param i: index of array to write to
        :type i: int
        """
        self.saved_properties[i, 0] = self.current_energy / self.total_spins
        for j, func in enumerate(self.propty_functions):
            self.saved_properties[i,j+1] = func(self.state)