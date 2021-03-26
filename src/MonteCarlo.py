import numpy as np


class MonteCarlo:
	def __init__(self, distributions, settings):
		self.distributions = distributions
		self.settings = settings
		self.size = settings["size"]
		self.dimensions = settings["dimensions"]

		self.grid = np.zeros(shape=(self.size, self.dimensions))