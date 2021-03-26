import numpy as np


class MetropolisAlgorithm:

    def __init__(self, distribution, settings):
        self.distribution = distribution
        self.settings = settings
        self.size = settings["size"]
        self.dimensions = settings["dimensions"]
        self.initial_distribution = settings["initial_distribution"]

        self.state = np.random.choice(a=[True, False], size=self.size**self.dimensions, p=[self.initial_distribution,1-self.initial_distribution])
        self.state = np.reshape(self.state, newshape=np.repeat(self.size, self.dimensions))
        
    def make_new_state(self):
        coordinate = np.random.randint(0,self.size,self.dimensions)
        self.new_state = np.copy(self.state)
        self.new_state[coordinate] = not self.new_state[coordinate]
    
    def simulation(self):
        make_new_state()
        if pobability_distribution(self.new_state) > pobability_distribution(self.state):