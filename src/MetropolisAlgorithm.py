import numpy as np


class MetropolisAlgorithm:
	def __init__(self, probability_distribution, settings):
		self.probability_distribution = probability_distribution
		self.settings = settings
		self.size = settings["size"]
		self.dimensions = settings["dimensions"]

		self.grid = np.zeros(shape=(self.size, self.dimensions))