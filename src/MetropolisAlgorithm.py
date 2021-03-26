import numpy as np


class MetropolisAlgorithm:

	def __init__(self, distributions, settings):
		self.distributions = distributions
		self.settings = settings
		self.size = settings["size"]
		self.dimensions = settings["dimensions"]
		self.initial_distribution = settings["initial_distribution"]

		self.grid = np.random.choice(a=[True, False], size=self.size**self.dimensions, p=self.initial_distribution)
		self.grid = np.reshape(self.grid, newshape=np.repeat(self.size, self.dimensions))