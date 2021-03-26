import numpy as np


class MetropolisAlgorithm:

    def __init__(self, pobability_distribution, settings):
        self.probability_distribution = pobability_distribution
        self.settings = settings
        self.size = settings["size"]
        self.dimensions = settings["dimensions"]
        self.initial_distribution = settings["initial_distribution"]

        self.state = np.random.choice(a=[True, False], size=self.size**self.dimensions, p=[self.initial_distribution,1-self.initial_distribution])
        self.state = np.reshape(self.state, newshape=np.repeat(self.size, self.dimensions))
        
    def make_new_state(self):
        coordinate = np.random.randint(0,self.size,self.dimensions)
        self.new_state = np.copy(self.state)
        self.new_state[coordinate] = ~ self.new_state[coordinate]
    
    def step(self):
        self.make_new_state()
        probability_current_state = self.probability_distribution(self.state)
        probability_new_state = self.probability_distribution(self.new_state)
        if probability_new_state > probability_current_state:
            acceptance_probability = 1
        else:
            acceptance_probability = probability_new_state /  probability_current_state
        self.state = np.random.choice(a=[self.state, self.new_state] , p=[1-acceptance_probability,acceptance_probability])