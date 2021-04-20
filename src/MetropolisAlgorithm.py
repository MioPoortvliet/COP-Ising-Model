import numpy as np
from src.utils import choice, Nchoice


class MetropolisAlgorithm:

    def __init__(self, model, property_functions, settings):
        self.model = model
        self.propty_functions = property_functions
        self.settings = settings
        self.size = settings["size"]
        self.dimensions = settings["dimensions"]
        self.initial_distribution = settings["initial_distribution"]

        #self.state = np.random.choice(a=[True, False], size=self.size**self.dimensions, p=[self.initial_distribution,1-self.initial_distribution])
        self.state = Nchoice(N=self.size ** self.dimensions, p=self.initial_distribution)

        self.state = np.reshape(self.state, newshape=np.repeat(self.size, self.dimensions))

        self.total_spins = self.state.size

        self.current_energy = self.model.hamiltonian(self.state)


    def make_new_state(self):
        coordinate = tuple(np.random.randint(0,self.size,self.dimensions))
        self.new_state = np.copy(self.state)
        self.new_state[coordinate] = not self.new_state[coordinate]

        return coordinate


    def step(self):
        coordinate = self.make_new_state()
        energy_difference = self.model.energy_difference(coordinate, self.new_state)

        if energy_difference < 0:
            acceptance_probability = 1
        else:
            acceptance_probability = self.model.hamiltonian_to_probability(energy_difference)

        #if np.random.choice(a=[True, False] , p=[acceptance_probability,1-acceptance_probability]):
        if choice(p=acceptance_probability):
            self.state = np.copy( self.new_state)
            self.current_energy += energy_difference


    def run_steps(self, steps):
        self.saved_properties = np.zeros(shape=(steps+1, len(self.propty_functions)+1))
        self.calc_properties(0)

        for i in range(steps):
            self.step()
            self.calc_properties(i+1)

        return self.saved_properties

    def calc_properties(self, i=0):
        self.saved_properties[i, 0] = self.current_energy / self.total_spins
        for j, func in enumerate(self.propty_functions):
            self.saved_properties[i,j+1] = func(self.state)